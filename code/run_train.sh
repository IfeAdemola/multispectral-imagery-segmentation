#!/usr/bin/bash

# Define default values for arguments
USE_MULTICLASS=false
USE_RGB=true
USE_RED=false
USE_NIR=false
USE_SWIR=false
USE_RED_EDGE=false
USE_SEASON=false
APPLY_AUGMENTATIONS=false
LEARNING_RATE=0.01
BATCH_SIZE=256
MAX_EPOCHS=2
ALPHA="0 0.9 0.1"
GAMMA=2
MODEL="unet"
PRETRAINED_BACKBONE=false
OUT_STRIDE=16
DATA_DIR_TRAIN="/home/k45848/multispectral-imagery-segmentation/data/processed/train"
DATA_DIR_VAL="/home/k45848/multispectral-imagery-segmentation/data/processed/test"
PROJECT_NAME="multispectral-imagery-segmentation"
SAVE_DIR="/home/k45848/multispectral-imagery-segmentation/runs"

# Parse command line arguments or update values accordingly
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_multiclass) USE_MULTICLASS=true ;;
        --use_rgb) USE_RGB=true ;;
        --use_red) USE_RED=true ;;
        --use_nir) USE_NIR=true ;;
        --use_swir) USE_SWIR=true ;;
        --use_red_edge) USE_RED_EDGE=true ;;
        --use_season) USE_SEASON=true ;;
        --apply_augmentations) APPLY_AUGMENTATIONS=true ;;  # Consistent with variable name
        --lr) LEARNING_RATE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --max_epochs) MAX_EPOCHS="$2"; shift ;;
        --alpha) ALPHA="$2"; shift ;;
        --gamma) GAMMA="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --pretrained_backbone) PRETRAINED_BACKBONE=true ;;
        --out_stride) OUT_STRIDE="$2"; shift ;;
        --data_dir_train) DATA_DIR_TRAIN="$2"; shift ;;
        --data_dir_val) DATA_DIR_VAL="$2"; shift ;;
        --project_name) PROJECT_NAME="$2"; shift ;;
        --save_dir) SAVE_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Display the set variables for confirmation
echo "Defined variables:"
echo "-------------------"
echo "USE_MULTICLASS: $USE_MULTICLASS"
echo "USE_RGB: $USE_RGB"
echo "USE_RED: $USE_RED"
echo "USE_NIR: $USE_NIR"
echo "USE_SWIR: $USE_SWIR"
echo "USE_RED_EDGE: $USE_RED_EDGE"
echo "USE_SEASON: $USE_SEASON"
echo "APPLY_AUGMENTATIONS: $APPLY_AUGMENTATIONS"  # Fixes display typo
echo "LEARNING_RATE: $LEARNING_RATE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_EPOCHS: $MAX_EPOCHS"
echo "ALPHA: $ALPHA"
echo "GAMMA: $GAMMA"
echo "MODEL: $MODEL"
echo "PRETRAINED_BACKBONE: $PRETRAINED_BACKBONE"
echo "OUT_STRIDE: $OUT_STRIDE"
echo "DATA_DIR_TRAIN: $DATA_DIR_TRAIN"
echo "DATA_DIR_VAL: $DATA_DIR_VAL"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "SAVE_DIR: $SAVE_DIR"
echo "-------------------"

# Construct the Python command dynamically
CMD="python3 train.py \
    --lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --gamma $GAMMA \
    --model $MODEL \
    --out_stride $OUT_STRIDE \
    --project_name \"$PROJECT_NAME\" \
    --save_dir \"$SAVE_DIR\""

# Append optional arguments only if they are set to true or non-empty
[ "$USE_MULTICLASS" = true ] && CMD+=" --use_multiclass"
[ "$USE_RGB" = true ] && CMD+=" --use_rgb"
[ "$USE_RED" = true ] && CMD+=" --use_red"
[ "$USE_NIR" = true ] && CMD+=" --use_nir"
[ "$USE_SWIR" = true ] && CMD+=" --use_swir"
[ "$USE_RED_EDGE" = true ] && CMD+=" --use_red_edge"
[ "$USE_SEASON" = true ] && CMD+=" --use_season"
[ "$APPLY_AUGMENTATIONS" = true ] && CMD+=" --apply_augmentations" 
[ "$PRETRAINED_BACKBONE" = true ] && CMD+=" --pretrained_backbone"
[ ! -z "$ALPHA" ] && CMD+=" --alpha $ALPHA"
[ ! -z "$DATA_DIR_TRAIN" ] && CMD+=" --data_dir_train \"$DATA_DIR_TRAIN\""
[ ! -z "$DATA_DIR_VAL" ] && CMD+=" --data_dir_val \"$DATA_DIR_VAL\""

# Display the final command for verification
echo "Running command: $CMD"
echo "-------------------"

# Run the constructed command
eval $CMD
