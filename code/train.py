import argparse
import random
import re
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             deeplabv3_resnet50)

import metrics
import utils
import wandb
from datasets import Forest, ToTensor
from loss.focal_loss import FocalLoss
from models.unet import UNet
from models.FCN import FullyConvNet
from argparse_config import get_args
from model_trainer import ModelTrainer
from torchgeo.models import FarSeg  # Example pretrained model on multispectral satellite data
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.models import ResNet50_Weights


def wandb_setup(args, model_name, selected_bands, ls, max_epochs, model, criterion, optimiser, save_dir):
    project_name = args.project_name
    name = f"{model_name}_{str(selected_bands)}_{ls}"
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # wandb initialisation
    wandb.init(project=project_name, 
               name=name,
               config=args)

    # model tracking
    wandb.watch(model, criterion, log="all")

    # model saving
    artifact = wandb.Artifact(name=re.sub(r'[^a-zA-Z0-9_.-]', '-', name), 
                              type="model",
                              description=f"Trained model for {args.model} at {current_datetime}",
                              metadata = {"epochs": max_epochs, "model": args.model,
                                           "batch_size": args.batch_size, "loss": criterion, 
                                           "optimiser": optimiser, "label": args.use_multiclass, 
                                            "bands": selected_bands}  
                            )
    
    # artifact.add_file

    # wandb.log_artifact(artifact)
    
def load_pretrained_model(args, model, model_name, num_classes, num_inputs):
    if args.model == 'deeplab': 
        model = "FarSeg(backbone='resnet50', num_classes)"
    elif args.model == 'unet':
        pretrained_weights = ResNet50_Weights.SENTINEL2_MI_RGB_SATLAS
        task = SemanticSegmentationTask(
            model="unet",
            backbone="resnet50",                 # ResNet50 backbone
            weights=pretrained_weights,          # Use Sentinel-2 pretrained weights
            in_channels=num_inputs,                       # Modify based on your input channels (Sentinel-2 RGB in this case)
            num_classes=num_classes,                       # Modify based on your classes (e.g., forest/deforested)
            num_filters=64,                      # Adjust the number of filters if needed
            freeze_backbone=True,                # Freeze the backbone for direct evaluation
            freeze_decoder=True,                 # Freeze the decoder as well for direct evaluation
            lr=0.001,                            # Learning rate (won't matter in this case since we aren't training)
            loss='ce',                           # Cross-entropy loss (irrelevant for direct evaluation)
        )
        model = task.model()

    # Freeze all layers for feature extraction (Direct Evaluation)
    for param in model.parameters():
        param.requires_grad = False

    return model

def load_pretrained_model_with_partial_freezing(model_name, num_classes, num_inputs):
    if model_name == 'deeplab':
        # Load DeepLab pretrained model from TorchGeo
        model = FarSeg(backbone='resnet50', num_classes=num_classes)

        # Modify input channels to match multispectral data
        model.backbone.conv1 = torch.nn.Conv2d(num_inputs, 64, 
                                               kernel_size=(7, 7), 
                                               stride=(2, 2), 
                                               padding=(3, 3), 
                                               bias=False)

        # Freeze backbone layers (i.e., ResNet50 backbone)
        for param in model.backbone.parameters():
            param.requires_grad = False  # Freeze these layers
        
    elif model_name == 'unet':
        # For UNet model, you can do similar freezing for its encoder backbone (if using a pretrained one)
        model = FarSeg(backbone='resnet50', num_classes=num_classes)

        # Freeze encoder backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model

def save_model_checkpoint():
    pass


def save_model(model, model_name, selected_bands, ls, save_dir):
    """
    Save the model locally
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trained_model_name = f"{model_name}_{str(selected_bands)}_{ls}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name, f"{trained_model_name}.pth")
    torch.save(model.state_dict(), model_path) 
    print(f"Model saved at {model_path}")


def main():
    # Load arguments
    args = get_args()
    print("="*20, "CONFIG", "="*20)
    for arg in vars(args):
        print('{0:20}  {1}'.format(arg, getattr(args, arg)))
    print()

    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("multi-gpu training not implemented! "
                                      + "try to run script as: "
                                      + "CUDA_VISIBLE_DEVICES=0 train.py")
    else:
        args.use_gpu = False

    model_name = args.model
    save_dir = args.save_dir

    transform = transforms.Compose([
        ToTensor(),
    ])

    # load datasets
    train_dataset = Forest(args.data_dir_train,
                           use_multiclass=args.use_multiclass,
                           transform=transform,
                           use_rgb=args.use_rgb,                           
                           use_red=args.use_red,
                           use_nir=args.use_nir,
                           use_red_edge=args.use_red_edge,
                           use_swir=args.use_swir,                       
                           use_season=args.use_season,
                           apply_augmentations=args.apply_augmentations)
    num_classes = train_dataset.num_classes
    num_inputs = train_dataset.n_inputs
    selected_bands = train_dataset.selected_bands

    print(f"Size of train data: {len(train_dataset)}")
    print(f"Train num_inputs: {num_inputs}")
    val_dataset = Forest(args.data_dir_val,
                           use_multiclass=args.use_multiclass,
                           transform=transform,
                           use_rgb=args.use_rgb,                           
                           use_red=args.use_red,
                           use_nir=args.use_nir,
                           use_red_edge=args.use_red_edge,
                           use_swir=args.use_swir, 
                           use_season=args.use_season)
    print(f"VAL num_inputs: {val_dataset.n_inputs}")
    print(f"VAL num_classses: {val_dataset.num_classes}")

    # set up dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    print(f'train dataset image size: {train_dataset[0][0].shape}')
    for X, Y in train_loader:
        print(f"images.shape: {X.shape}")
        print(f"labels.shape: {Y.shape}")
        break

    # set up network
    if args.model == "deeplab":
        weights_backbone = ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=None, 
                                   weights_backbone=weights_backbone, 
                                   num_classes=num_classes)
        model.backbone.conv1 = torch.nn.Conv2d(num_inputs, 64, 
                                               kernel_size=(7, 7), 
                                               stride=(2, 2), 
                                               padding=(3, 3), 
                                               bias=False)
    elif args.model == 'unet':
        model = UNet(n_classes=num_classes,
                     n_channels=num_inputs)
    elif args.model == 'fcn':
        model = FullyConvNet(input_channels=num_inputs,
                    num_classes=num_classes)
        
    if args.use_gpu:
        model = model.cuda()

    # define number of epochs
    max_epochs = args.max_epochs 

    # define label structure
    use_multiclass = args.use_multiclass
    ls = 1 if use_multiclass else 0

    if args.alpha and len(args.alpha) == num_classes:
        alpha = torch.tensor(args.alpha, dtype=torch.float32).cuda()
    else:
        raise ValueError(f"The length of alpha must be {num_classes} for the segmentation map.")


    finetune = False
    if finetune:
        # pretrained model
        model = load_pretrained_model(args, model, model_name, num_classes, num_inputs)
        finetune_lr = args.lr*0.1
        optimiser = torch.optim.Adam(model.parameters(), lr=finetune_lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_epochs)
    else:
        # Create the focal loss criterion and the optimiser
        criterion = FocalLoss(alpha=alpha, gamma=args.gamma, ignore_index=0) # gamma=0: Crossentropyloss
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_epochs)

    

    # setup wandb
    wandb_setup(args, model_name, selected_bands, ls, max_epochs, model, criterion, optimiser, save_dir)
    # train network 
    trainer = ModelTrainer(args)
    model = trainer.train(use_multiclass, model, train_loader, val_loader, criterion, optimiser, scheduler, max_epochs)
    
    
  
if __name__ == "__main__":
    main()
    

