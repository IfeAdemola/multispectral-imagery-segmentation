import argparse

def get_args():
    # define and parse arguments
    parser = argparse.ArgumentParser(description="Train a model for semnatic segmentation")

    #input/output
    parser.add_argument('--use_multiclass', action='store_true', default=False,
                        help='complexity of classes; True if labelling is more specific (6 classes), else False (3 classes)')
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
                        help='use weather season band')
                        
    # training hyperparameters
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for training and validation \
                              (default: 256)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')
    
    # loss arguments
    # alpha and gamma arguments
    parser.add_argument('--alpha', type=float, nargs='+', default=None, help='Class weights (alpha) for focal loss, provide as space-separated values (e.g. --alpha 0 0.03 0.26 0.48 0.21 0.002)')
    parser.add_argument('--gamma', type=float, default=2, help='Gamma value for Focal Loss (default: 2)')

    # network
    parser.add_argument('--model', type=str, choices=['deeplabv3+', 'unet'],
                        default='unet',
                        help="network architecture (default: unet)")
#     parser.add_argument('--use_gpu', type=bool, default=True,
#                         help='use gpu for training')
    
    #deeplab-specific
    parser.add_argument('--pretrained_backbone', action='store_true',
                        default=False,
                        help='initialize ResNet-101 backbone with ImageNet \
                              pre-trained weights')
    parser.add_argument('--out_stride', type=int, choices=[8, 16], default=16,
                        help='network output stride (default: 16)')

    # training strategy
    parser.add_argument('--train_mode', type=str, default='scratch', choices=['scratch', 'pretrained'],
                        help='Whether to train from scratch or use a pretrained strategy')
    parser.add_argument('--pretrained_strategy', type=str, default='fine_tune_all_layers',
                        choices=['direct_evaluation', 'fine_tune_all_layers', 'fine_tune_last_layers'],
                        help='Pretrained strategy to use if train_mode is pretrained')

    
    # data
    parser.add_argument('--data_dir_train', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--data_dir_val', type=str, default=None,
                        help='path to validation dataset')
    parser.add_argument('--apply_augmentations', action='store_true', default=False,
                        help='apply augmentation to train dataset')
    
    # monitoring and model storage
    parser.add_argument('--project_name', type=str, default="multispectral-imagery-segmentation",
                        help='wandb project name')
    parser.add_argument('--save_dir', type=str, default="./trained_models",
                        help='path to save model')
    

    return parser.parse_args()


