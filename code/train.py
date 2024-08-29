import argparse
import os
import random
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             deeplabv3_resnet50)
from tqdm import tqdm

import metrics
import utils
import wandb
from datasets import Forest, ToTensor
from loss.focal_loss import FocalLoss
from models.unet import UNet


class ModelTrainer:
        
    def __init__(self, args):
        self.args = args

    def train(self, label_structure, model, train_loader, val_loader,
              criterion, optimiser, max_epochs, do_reweighting=False, 
              high_conf_weight=1.0, low_conf_weight=0.5, structure_size=3):  
        
        # Initialize the scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=0)
        
        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, criterion, log="all", log_freq=10)

        for epoch in range(max_epochs):
            print("="*20, "EPOCH", epoch + 1, "/", str(max_epochs), "="*20)

            model.train()
            epoch_loss = 0
            train_conf_mat = metrics.ConfusionMatrix(train_loader.dataset.num_classes)


            with tqdm(train_loader, desc="[Train]", unit='batch') as pbar:
                for batch_idx, (images, labels) in enumerate(pbar):
                    
                    if self.args.use_gpu:
                        images, labels = images.cuda(), labels.cuda()

                    
                    "no longer necessary: criterion (focal loss or not already covers this)"
                    # Create masks based on your criteria 
                    if do_reweighting:
                        # Use the custom reweighting mask
                        mask = self.create_weight_mask(labels, high_conf_weight, low_conf_weight, structure_size)

                    # reset gradients
                    optimiser.zero_grad()

                    # forward pass
                    outputs = model(images)
                    if isinstance(outputs, dict) and 'out' in outputs:
                        outputs = outputs['out']

                    # Calculate raw loss
                    loss = criterion(outputs, labels)  # Batch Per-pixel loss

                    # Backward pass and optimize
                    loss.backward()
                    optimiser.step()

                    preds = torch.argmax(outputs, dim=1)
                    epoch_loss += loss.item() * images.size(0)

                    train_conf_mat.add_batch(labels, preds)

                    # Update progress bar
                    pbar.set_postfix(loss=epoch_loss/(pbar.n+1) * train_loader.batch_size)
                    pbar.set_description("[Train] Loss: {:.4f}".format(
                    round(loss.item(), 4)))       
                    
            # Calculate the loss and IoU per epoch
            average_epoch_loss = epoch_loss / len(train_loader.dataset)            
            train_iou = train_conf_mat.get_IoU() # list of iou for each class (per epoch)
            train_miou = train_conf_mat.get_mIoU() # float mean iou per epoch
            print(f'[Train] Average epoch loss: {average_epoch_loss:.4f}')

            # pbar.close()

            # Log metrices and loss
            class_names = utils.classnames(label_structure)[1:]
            train_metrics = {f"Train IoU {class_name}": iou for class_name, iou in zip(class_names, train_iou)}
            train_metrics["Train overall IoU"] = train_miou
            train_metrics["Train loss"] = average_epoch_loss
            train_metrics["epoch"] = epoch + 1
            wandb.log(train_metrics) 

            # Update the scheduler
            scheduler.step()

            # run validation
            self.val(label_structure, model, val_loader, criterion, epoch)
                       
        return model
    
    def val(self, label_structure, model, dataloader, criterion, epoch):

        # set model to evaluation mode
        model.eval()
        val_loss = 0
        conf_mat = metrics.ConfusionMatrix(dataloader.dataset.num_classes)

        with torch.no_grad(): # disable gradient calculation
            with tqdm(dataloader, desc="[Val]", unit="batch") as pbar:
                for idx, (images, labels) in enumerate(pbar):

                    if self.args.use_gpu:
                        images, labels = images.cuda(), labels.cuda()

                    # Forward pass
                    outputs = model(images)
                    if isinstance(outputs, dict) and 'out' in outputs:
                        outputs = outputs['out']

                    # Loss calculation
                    loss = criterion(outputs, labels) #batch_loss

                    preds = torch.argmax(outputs, dim=1)
                    val_loss += loss.item()*images.size(0)  #changed from masked_loss to loss

                    # calculate error metrics
                    conf_mat.add_batch(labels, preds)
                    aa = conf_mat.get_aa()
                    miou = conf_mat.get_mIoU()

                    # Update progress bar
                    pbar.set_postfix(loss=val_loss/(pbar.n+1) * dataloader.batch_size) #loss=f"{batch_loss.item():.4f}"
                    pbar.set_description("[Val] AA: {:.2f}%, IoU: {:.2f}%".format(aa , miou * 100))            

    
                pbar.close()
                
                # Average loss and IoU over the evaluation dataset
                avg_epoch_loss = val_loss / len(dataloader.dataset)
                val_iou = conf_mat.get_IoU()
                val_miou = conf_mat.get_mIoU()
                print(f'[Val] Average epoch loss: {avg_epoch_loss:.4f}')

                # Log metrices and loss
                # class_names = utils.classnames()
                class_names = utils.classnames(label_structure)[1:]
                val_metrics = {f"Val IoU {class_name}": iou for class_name, iou in zip(class_names, val_iou)}
                val_metrics["Val overall IoU"] = val_miou
                val_metrics["Val loss"] = avg_epoch_loss
                val_metrics["epoch"] = epoch + 1
                wandb.log(val_metrics)     

                # Log input image, groundtruth and prediction
                if (epoch +1) % 50 == 0:
                    random_idx = np.random.randint(preds.size(0))
                    in_log = images[random_idx, ...]
                    in_log = in_log.permute(1,2,0)
                    input_log = utils.display_input(in_log.cpu().numpy())
                    gt_log = utils.display_label(labels[random_idx,...].cpu().numpy(), label_structure)

                    #validity mask (0: invalid 1:valid). Just for display so that it would still show class 0
                    pred_display_mask = (labels[random_idx, ...].cpu().numpy() != 0).astype(np.uint8)
                    masked_pred = preds[random_idx, ...].cpu().numpy() * pred_display_mask
                    # Display the masked predictions
                    masked_pred_log = utils.display_label(masked_pred, label_structure)
                    # pred_log = utils.display_label(preds[random_idx,...].cpu().numpy(), label_structure)
                    images_log = [input_log, gt_log, masked_pred_log]

                    wandb.log({"Images": [wandb.Image(i) for i in images_log]})
                
                model.train()
                    
    def create_weight_mask(self, labels, high_conf_weight=1.0, low_conf_weight=0.5, structure_size=3):
        """
        Creates a weight mask for reweighting loss based on the confidence of pixel labels.
        Only considers classes 1 (brown) and 2 (green).
        
        Parameters:
            labels (Tensor): The ground truth segmentation mask.
            high_conf_weight (float): Weight for high-confidence pixels.
            low_conf_weight (float): Weight for low-confidence border pixels.
            structure_size (int): Size of the structure used for erosion to identify high-confidence pixels.

        Returns:
            weight_mask (Tensor): A mask with the same size as labels, with weights applied.
        """
        # Convert the labels to a numpy array
        labels_np = labels.cpu().numpy()

        # Initialize weight mask with ones for class 2 (green) and zero for class 0 (grey)
        weight_mask = torch.ones_like(labels, dtype=torch.float32)
        weight_mask[labels == 0] = 0  # Ignore class 0 (grey)

        # Apply high and low confidence weights to class 1 (brown) only
        class_1_mask = (labels_np == 1)
        high_conf_mask = binary_erosion(class_1_mask, structure=np.ones((structure_size, structure_size)))

        # Convert back to tensor and update the weight mask
        weight_mask[labels == 1] = low_conf_weight  # Low confidence weight for borders
        weight_mask[torch.tensor(high_conf_mask)] = high_conf_weight  # High confidence weight for inner pixels

        return weight_mask


    def seed():
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        random.seed(hash("setting random seeds") % 2**32 - 1)
        np.random.seed(hash("improves reproducibility") % 2**32 - 1)
        torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)



def main():

    # define and parse arguments
    parser = argparse.ArgumentParser()

    #input/output
    parser.add_argument('--label_structure', action='store_true', default=False,
                        help='labelling structure; True if labelling is more specific (6 classes), else False (3 classes)')
    parser.add_argument('--use_rgb', action='store_true', default=False,
                        help='use sentinel-2 rgb bands')
    parser.add_argument('--use_red', action='store_true', default=False,
                        help='use sentinel-2 red bands')
    parser.add_argument('--use_nir', action='store_true', default=False,
                        help='use sentinel-2 near infrared (20 m) bands')
    parser.add_argument('--use_swir', action='store_true', default=False,
                        help='use sentinel-2 SWIR bands') 
    parser.add_argument('--use_red_edge', action='store_true', default=False,
                        help='use sentinel-2 red edge bands')
    parser.add_argument('--use_season', action='store_true', default=False,
                        help='use weather season high-resolution bands')
                        
    # training hyperparameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training and validation \
                              (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='number of training epochs (default: 50)')
    
    # network
    parser.add_argument('--model', type=str, choices=['deeplab', 'unet'],
                        default='deeplab',
                        help="network architecture (default: deeplab)")
    
    #deeplab-specific

    # data
    parser.add_argument('--data_dir_train', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--data_dir_val', type=str, default=None,
                        help='path to validation dataset')
    
    args = parser.parse_args()
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

    transform = transforms.Compose([
        ToTensor(),
    ])
    # load datasets
    train_dataset = Forest(args.data_dir_train,
                           label_structure=args.label_structure,
                           transform=transform,
                           use_rgb=args.use_rgb,                           
                           use_red=args.use_red,
                           use_nir=args.use_nir,
                           use_red_edge=args.use_red_edge,
                           use_swir=args.use_swir,                       
                           use_season=args.use_season)
    num_classes = train_dataset.num_classes
    num_inputs = train_dataset.n_inputs
    selected_bands = train_dataset.selected_bands

    print(f"Size of train data: {len(train_dataset)}")
    print(f"Train num_inputs: {num_inputs}")
    val_dataset = Forest(args.data_dir_val,
                           label_structure=args.label_structure,
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
        

    if args.use_gpu:
        model = model.cuda()

    # define number of epochs
    max_epochs = args.max_epochs 

    # define label structure
    label_structure = args.label_structure
    if label_structure:
        # extended segmentation map
        ls = 1   
        # Define class weights (as in the weighted focal loss)
        alpha = torch.tensor([0, 0.03877981, 0.26108468, 0.48395104, 0.21362894, 0.00255552], dtype=torch.float32).cuda()
    else:
        # segmentation map
        ls = 0   
        alpha = torch.tensor([0.0, 0.915, 0.085], dtype=torch.float32).cuda()
    
    
    
    # Create the focal loss criterion
    criterion = FocalLoss(alpha=alpha, gamma=2, ignore_index=0) # gamma=0: Crossentropyloss
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)


    """Cleaner way to set up wandb"""
    # set up weight and biases logging
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format current datetime
    
    run_name = f"{args.model}_{str(selected_bands)}_{ls}"  #_wsss
    
    print("Attempting to initialize W&B...")
    wandb.init(project="complex_supervised_29.08", 
                     name=run_name, 
                     config=args)  # Initialize W&B

    # train network
    trainer = ModelTrainer(args)
    model = trainer.train(label_structure, model, train_loader, val_loader, criterion, optimiser, max_epochs)

    # Save the trained model locally
    model_save_path =  f"/home/k45848/multispectral-imagery-segmentation/trained_models/{args.model}_{str(selected_bands)}_{ls}.pth" #f"{args.model}_{current_datetime}.pth"
    torch.save(model.state_dict(), model_save_path) 

    # Create and log a W&B artifact
    name=f"{args.model}_{str(selected_bands)}_{ls}"  #_wss_artifact
    artifact = wandb.Artifact(
        name = re.sub(r'[^a-zA-Z0-9_.-]', '-', name),
        type="model",
        description=f"Trained model for {args.model} at {current_datetime}",
        metadata={"epochs": max_epochs, "model": args.model, "batch_size": args.batch_size, "loss": criterion, "optimiser": optimiser,
                   "label": args.label_structure, "bands": selected_bands}  # Add any additional metadata here
    )

    # Add the local model file to the artifact
    artifact.add_file(model_save_path)

    # Log the artifact to W&B
    wandb.log_artifact(artifact)

    wandb.finish()
        

if __name__ == "__main__":
    main()



# python train.py --label --use_red --use_nir --lr 0.001 --batch_size 512 --max_epochs 600 --model unet --data_dir_train "/home/k45848/multispectral-imagery-segmentation/data/19.08/train" --data_dir_val "/home/k45848/multispectral-imagery-segmentation/data/19.08/val"
