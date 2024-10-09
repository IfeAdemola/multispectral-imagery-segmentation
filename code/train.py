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
from torchvision.ops import sigmoid_focal_loss
# from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             deeplabv3_resnet50)

import metrics
import utils
import wandb
import torch.nn as nn
from datasets import Forest, ToTensor
from loss.focal_loss import FocalLoss
from models.unet import UNet
from models.FCN import FullyConvNet
from argparse_config import get_args
from model_trainer import ModelTrainer
from train_model import TrainModel
from torchgeo.models import FarSeg  # Example pretrained model on multispectral satellite data
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.models import ResNet50_Weights, ResNet18_Weights


def wandb_setup(args, model_name, selected_bands, ls, max_epochs, model, criterion, optimiser, pretrained_weights, save_dir):
    project_name = args.project_name
    if args.train_mode == "scratch":
        name = f"{model_name}_{str(selected_bands)}_{ls}_{args.train_mode}"
    else:
        name = f"{model_name}_{str(selected_bands)}_{ls}_{args.pretrained_strategy}"
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
                                            "bands": selected_bands,
                                            "pretrained_weights": str(pretrained_weights)}  
                            )
    
    artifact.add_file('best_model.pth')

    wandb.log_artifact(artifact)
    
# def direct_evaluation(args, num_classes, num_inputs, pretrained_weights):
#     task = SemanticSegmentationTask(
#         model=args.model,
#         backbone="resnet50",                
#         weights=pretrained_weights,          
#         in_channels=num_inputs,                       
#         num_classes=num_classes,                      
#         num_filters=64,                    
#         freeze_backbone=True,                
#         freeze_decoder=True,
#     )
#     task.model.eval()

#     # Freeze all layers
#     for param in task.model.parameters():
#         param.requires_grad = False

#     return task.model 

# def fine_tune_all_layers(args, num_classes, num_inputs, pretrained_weights, class_weights, lr):
#     task = SemanticSegmentationTask(
#         model=args.model,
#         backbone="resnet50",                
#         weights=pretrained_weights,         
#         in_channels=num_inputs,              
#         num_classes=num_classes,  
#         class_weights= class_weights,           
#         freeze_backbone=False,               
#         freeze_decoder=False,                
#         lr=lr,                               
#         loss='focal',                           
#     )
#     return task.model

# def fine_tune_last_layers(args, num_classes, num_inputs, pretrained_weights, class_weights, lr):

#     task = SemanticSegmentationTask(
#         model=args.model,
#         backbone="resnet50",                 
#         weights=pretrained_weights,          
#         in_channels=num_inputs,                       
#         num_classes=num_classes,    
#         class_weights= class_weights,                    
#         freeze_backbone=True,                
#         freeze_decoder=False,                 
#         lr=lr,                            
#         loss='focal',                           
#     )

#     # Freeze encoder backbone
#     # for param in model.backbone.parameters():
#     #     param.requires_grad = False
    
#     # model.encoder.conv1 = torch.nn.Conv2d(
#     #         num_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#     #     )

#     return task.model

def pretrained_strategy(args, num_classes, num_inputs, pretrained_weights, class_weights, lr):
    assert args.train_mode == "pretrained" , "Training mode must be set to 'pretrained'."
    if args.pretrained_strategy == 'direct_evaluation':
        freeze_backbone=True,                
        freeze_decoder=True
    elif args.pretrained_strategy == 'fine_tune_all_layers':
        freeze_backbone=False,
        freeze_decoder=False
    elif args.pretrained_strategy == 'fine_tune_last_layers':
        freeze_backbone=True,
        freeze_decoder=False
    else:
        raise ValueError("Invalid pretrained strategy. Select one of 'direct_evaluation', 'fine_tune_all_layers', 'fine_tune_last_layers' ")
    
    task = SemanticSegmentationTask(
        model=args.model,
        backbone="resnet18",                 
        weights=pretrained_weights,          
        in_channels=num_inputs,                       
        num_classes=num_classes,    
        class_weights= class_weights,                    
        freeze_backbone=freeze_backbone,                
        freeze_decoder=freeze_decoder,                 
        lr=lr,                            
        loss='focal',                           
    )

    if args.pretrained_strategy == 'direct_evaluation':
        task.model.eval()
        for param in task.model.parameters():
            param.requires_grad = False

    return task.model
    
def initialise_model(args, pretrained_weights, num_classes, num_inputs, class_weights, lr):
    # If training from scratch
    if args.train_mode == "scratch":
        task = SemanticSegmentationTask(
            model=args.model,
            backbone="resnet18",                
            weights=None,  # No pretrained weights
            in_channels=num_inputs,   
            num_classes=num_classes,
            class_weights=class_weights,
            lr=lr,
            loss='focal'   

        )
        model = task.model
    
    # If using pretrained weights strategy
    elif args.train_mode == "pretrained":
        # Use the new pretrained_strategy function
        model = pretrained_strategy(
            args=args,
            num_classes=num_classes,
            num_inputs=num_inputs,
            pretrained_weights=pretrained_weights,
            class_weights=class_weights,
            lr=args.lr
        )

    # Move the model to GPU if available
    if args.use_gpu:
        model = model.cuda()

    return model


# def initialise_model(args, pretrained_weights, num_classes, num_inputs, class_weights):
#     if args.train_mode == "scratch":
#         if args.model == 'unet':
#             model = UNet(n_classes=num_classes,
#                         n_channels=num_inputs)
#         elif args.model == 'deeplabv3':
#             model = deeplabv3_resnet50(weights=pretrained_weights)
#             model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
#     elif args.train_mode == "pretrained":
#         model = pretrained_strategy(args, num_classes, num_inputs, pretrained_weights, class_weights, args.lr)
    # return model

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
     

    # define number of epochs
    max_epochs = args.max_epochs 

    # define label structure
    use_multiclass = args.use_multiclass
    ls = 1 if use_multiclass else 0

    print(f"args.alpha: {args.alpha}")
    if args.alpha and len(args.alpha) == num_classes:
        alpha = torch.tensor(args.alpha, dtype=torch.float32).cuda()
    else:
        raise ValueError(f"The length of alpha must be {num_classes} for the segmentation map.")
    print(f"alpha: {alpha}")

    if num_inputs == 3:
        pretrained_weights = ResNet18_Weights.SENTINEL2_RGB_SECO #ResNet18_Weights.SENTINEL2_RGB_MOCO
    elif num_inputs == 10:
        pretrained_weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

    # model = initialise_model(args, pretrained_weights, num_classes, num_inputs, alpha, args.lr)
    model = UNet(n_classes=num_classes, n_channels=num_inputs)
    model = model.cuda()

    # Set up other training components (criterion, optimiser, scheduler)
    # criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss(weight=alpha)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_epochs)  

    # setup wandb
    wandb_setup(args, model_name, selected_bands, ls, max_epochs, model, criterion, optimiser, pretrained_weights, save_dir)
    # train network 
    trainer = ModelTrainer(args)
    model = trainer.train(use_multiclass, model, train_loader, val_loader, criterion, optimiser, scheduler, max_epochs)
    
    
  
if __name__ == "__main__":
    main()
    

