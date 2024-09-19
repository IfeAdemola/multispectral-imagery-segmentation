import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

import metrics
import utils
import wandb


class ModelTrainer:     
    def __init__(self, args):
        self.args = args

    def train(self, use_multiclass, model, train_loader, val_loader,
              criterion, optimiser, scheduler, max_epochs, do_reweighting=False, 
              high_conf_weight=1.0, low_conf_weight=0.5, structure_size=3):  
               
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
            class_names = utils.classnames(use_multiclass)[1:]
            train_metrics = {f"Train IoU {class_name}": iou for class_name, iou in zip(class_names, train_iou)}
            train_metrics["Train overall IoU"] = train_miou
            train_metrics["Train loss"] = average_epoch_loss
            train_metrics["epoch"] = epoch + 1
            wandb.log(train_metrics) 

            # Update the scheduler
            scheduler.step()

            # run validation
            self.val(use_multiclass, model, val_loader, criterion, epoch, max_epochs)
                       
        return model
    
    def val(self, use_multiclass, model, dataloader, criterion, epoch, max_epochs):

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
                class_names = utils.classnames(use_multiclass)[1:]
                val_metrics = {f"Val IoU {class_name}": iou for class_name, iou in zip(class_names, val_iou)}
                val_metrics["Val overall IoU"] = val_miou
                val_metrics["Val loss"] = avg_epoch_loss
                val_metrics["epoch"] = epoch + 1
                wandb.log(val_metrics)     

                # Log the confusion matrix only after the final epoch
                if epoch + 1 == max_epochs:
                    confusion_matrix = conf_mat.norm_on_lines()
                    wandb.log({
                        "final_confusion_matrix": wandb.Table(
                            columns=["Predicted"] + class_names,
                            data=[[class_names[i]] + row.tolist() for i, row in enumerate(confusion_matrix)]
                        )
                    })

                # # Log input image, groundtruth and prediction
                if (epoch +1) % 10 == 0:
                    random_idx = np.random.randint(preds.size(0))
                    in_log = images[random_idx, ...]
                    in_log = in_log.permute(1,2,0)
                    input_log = utils.display_input(in_log.cpu().numpy())
                    gt_log = utils.display_label(labels[random_idx,...].cpu().numpy(), use_multiclass)

                    #validity mask (0: invalid 1:valid). Just for display so that it would still show class 0
                    pred_display_mask = (labels[random_idx, ...].cpu().numpy() != 0).astype(np.uint8)
                    masked_pred = preds[random_idx, ...].cpu().numpy() * pred_display_mask
                    # Display the masked predictions
                    masked_pred_log = utils.display_label(masked_pred, use_multiclass)
                    # pred_log = utils.display_label(preds[random_idx,...].cpu().numpy(), use_multiclass)
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
