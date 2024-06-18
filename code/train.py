import numpy as np
import os
import torch
import torch.optim as optim
import argparse
import random
import utils
import wandb
import metrics
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets import Forest, ToTensor
from models.unet import UNet
from datetime import datetime

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import ResNet50_Weights


class ModelTrainer:
        
    def __init__(self, args):
        self.args = args

    def train(self, label_structure, model, train_loader, val_loader,
              criterion, optimiser, max_epochs, step=0):
        
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

                    # Create masks based on your criteria (example: ignore classes 0 and 2)
                    mask = (labels != 0).float()

                    # reset gradients
                    optimiser.zero_grad()

                    # forward pass
                    outputs = model(images)
                    if isinstance(outputs, dict) and 'out' in outputs:
                        outputs = outputs['out']

                    # calculate loss
                    loss = criterion(outputs, labels)  #batch_loss
                    masked_loss = (loss*mask).sum()/mask.sum()

                    # backward pass and optimize
                    masked_loss.backward()
                    optimiser.step()

                    preds = torch.argmax(outputs, dim=1)
                    epoch_loss += masked_loss.item() * images.size(0)

                    train_conf_mat.add_batch(labels, preds)


                    # Update progress bar
                    pbar.set_postfix(loss=epoch_loss/(pbar.n+1) * train_loader.batch_size)
                    pbar.set_description("[Train] Loss: {:.4f}".format(
                    round(masked_loss.item(), 4)))       
                    
            # Calculate the loss and IoU per epoch
            average_epoch_loss = epoch_loss / len(train_loader.dataset)            
            train_iou = train_conf_mat.get_IoU() # list of iou for each class (per epoch)
            train_miou = train_conf_mat.get_mIoU() # float mean iou per epoch
            print(f'[Train] Average epoch loss: {average_epoch_loss:.4f}')

            pbar.close()

            # Log metrices and loss
            # class_names = utils.classnames()
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

        # main validation loop
        epoch_loss = 0
        conf_mat = metrics.ConfusionMatrix(dataloader.dataset.num_classes)
        with tqdm(dataloader, desc="[Val]", unit="batch") as pbar:
            for idx, (images, labels) in enumerate(pbar):

                if self.args.use_gpu:
                    images, labels = images.cuda(), labels.cuda()

                # Create masks based on your criteria (example: ignore classes 0 and 2)
                    mask = (labels != 0).float()

                # Forward pass
                with torch.no_grad():
                    outputs = model(images)
                    if isinstance(outputs, dict) and 'out' in outputs:
                        outputs = outputs['out']

                # Loss calculation
                loss = criterion(outputs, labels) #batch_loss
                masked_loss = (loss*mask).sum()/mask.sum()
                
                preds = torch.argmax(outputs, dim=1)
                epoch_loss += masked_loss.item()*images.size(0)

                # calculate error metrics
                conf_mat.add_batch(labels, preds)
                aa = conf_mat.get_aa()
                miou = conf_mat.get_mIoU()

                # Update progress bar
                pbar.set_postfix(loss=epoch_loss/(pbar.n+1) * dataloader.batch_size) #loss=f"{batch_loss.item():.4f}"
                pbar.set_description("[Val] AA: {:.2f}%, IoU: {:.2f}%".format(aa * 100, miou * 100))            

 
            pbar.close()
            
            # Average loss and IoU over the evaluation dataset
            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
            val_iou = conf_mat.get_IoU()
            val_miou = conf_mat.get_mIoU()
            print(f'[Val] Average epoch loss: {avg_epoch_loss:.4f}')

            # Log metrices and loss
            # class_names = utils.classnames()
            class_names = utils.classnames(label_structure)[1:]
            print(class_names)
            val_metrics = {f"Val IoU {class_name}": iou for class_name, iou in zip(class_names, val_iou)}
            val_metrics["Val overall IoU"] = val_miou
            val_metrics["Val loss"] = avg_epoch_loss
            val_metrics["epoch"] = epoch + 1
            wandb.log(val_metrics)     

            # Log input image, groundtruth and prediction
            if (epoch +1) % 10 == 0:
                random_idx = np.random.randint(preds.size(0))
                in_log = images[random_idx, ...]
                in_log = in_log.permute(1,2,0)
                input_log = utils.display_input(in_log.cpu().numpy())
                pred_log = utils.display_label(preds[random_idx,...].cpu().numpy(), label_structure)
                gt_log = utils.display_label(labels[random_idx,...].cpu().numpy(), label_structure)
                images_log = [input_log, gt_log, pred_log]

                wandb.log({"Images": [wandb.Image(i) for i in images_log]})
               
            model.train()

        
    def export_model(self, model, optimiser=None, name=None, step=None):

        if name is not None:
            out_file = name
        else:
            out_file = "checkpoint"
            if step is not None:
                out_file += "_step_" + str(step)
        out_file = os.path.join(self.args.checkpoint_dir, out_file + ".pth")

        # save model
        data = {"model_state_dict": model.state_dict()}
        if step is not None:
            data["step"] = step
        if optimiser is not None:
            data["optimiser_state_dict"]= optimiser.state_dict()
        torch.save()

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
    parser.add_argument('--use_lr', action='store_true', default=False,
                        help='use sentinel-2 low-resolution (10 m) bands')
    parser.add_argument('--use_mr', action='store_true', default=False,
                        help='use sentinel-2 medium-resolution (20 m) bands')
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
                           use_lr=args.use_lr,
                           use_mr=args.use_mr,
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
                           use_lr=args.use_lr,
                           use_mr=args.use_mr,
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

    # class weights
    # class_weights = torch.tensor([0,0.9,0.1]).cuda()
    # define loss function
    criterion = nn.CrossEntropyLoss(reduction="none")   #weight=class_weights  # ignore_index=0,  weights=torch.tensor([0.1,0.2,0.7]) # can include weights if ds is unbalanced
    
    # set up optimiser
    # optimiser = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # set up weight and biases logging
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format current datetime
    if label_structure:
        ls = 1   # extended segmentation map
    else:
        ls = 0   # segmentation map
    run_name = f"{args.model}_{str(selected_bands)}_{ls}"
    
    print("Attempting to initialize W&B...")
    wandb.init(project="Thesis_experiments", 
                     name=run_name, 
                     config=args)  # Initialize W&B

    # train network
    trainer = ModelTrainer(args)
    model = trainer.train(label_structure, model, train_loader, val_loader, criterion, optimiser, max_epochs)

    # Save the trained model locally
    model_save_path =  f"/home/k45848/multispectral-imagery-segmentation/models/{args.model}_{str(selected_bands)}_{ls}.pth" #f"{args.model}_{current_datetime}.pth"
    torch.save(model.state_dict(), model_save_path) 

    # Create and log a W&B artifact
    artifact = wandb.Artifact(
        name=f"{args.model}_{str(selected_bands)}_{ls}_artifact",
        type="model",
        description=f"Trained model for {args.model} at {current_datetime}",
        metadata={"epochs": max_epochs, "model": args.model, "batch_size": args.batch_size, "loss": criterion, "optimiser": optimiser, "label": args.label_structure, "bands": selected_bands}  # Add any additional metadata here
    )

    # Add the local model file to the artifact
    artifact.add_file(model_save_path)

    # Log the artifact to W&B
    wandb.log_artifact(artifact)

    wandb.finish()
        

if __name__ == "__main__":
    main()



# python train.py --label --use_rgb --use_lr --use_mr --use_season --lr 0.001 --batch_size 64 --max_epochs 2 --model deeplab --data_dir_train "/home/k45848/multispectral-imagery-segmentation/data/31.05/train" --data_dir_val "/home/k45848/multispectral-imagery-segmentation/data/31.05/eval"
# python train.py --use_rgb --use_lr --use_season --lr 0.001 --batch_size 64 --max_epochs 200 --model unet --data_dir_train "/home/k45848/multispectral-imagery-segmentation/data/08.06/train" --data_dir_val "/home/k45848/multispectral-imagery-segmentation/data/08.06/val"
# python train.py --label --use_rgb --lr 0.001 --batch_size 64 --max_epochs 200 --model unet --data_dir_train "/home/k45848/multispectral-imagery-segmentation/data/08.06/train" --data_dir_val "/home/k45848/multispectral-imagery-segmentation/data/08.06/val"

