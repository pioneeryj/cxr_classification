#!/bin/bash
# General training script for CXR classification
# Edit the variables below

set -e

##############################################
# EDIT THESE SETTINGS
##############################################

# CONFIG_FILE="configs/medclip_config.yaml"
# CONFIG_FILE="configs/biovil_config.yaml"
# CONFIG_FILE="configs/medklip_config.yaml"
CONFIG_FILE="configs/biomedclip_config.yaml"
#CONFIG_FILE="configs/resnet50_config.yaml"

# Set GPU (change if needed)
GPU_ID=0

# Checkpoint directory (leave empty to auto-generate: {model}_posweight={}_resampling={})
#CHECKPOINT_DIR="/mnt/HDD/yoonji/mrg/cxr_classification/weight/resnet50_posweight=True_resampling=True"
CHECKPOINT_DIR="/mnt/HDD/yoonji/mrg/cxr_classification/weight/BiomedCLIP_posweight=True_resampling=True_lora_r=16"
# Resume training from checkpoint? (true/false)
RESUME="true"

# Class imbalance handling
POS_WEIGHT="true"           # Use pos_weight in BCEWithLogitsLoss (true/false)
SQRT_RESAMPLING="true"     # Use square root resampling sampler (true/false)

##############################################

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create checkpoint directory if specified
if [ -n "$CHECKPOINT_DIR" ]; then
    mkdir -p "$CHECKPOINT_DIR"
fi

echo "=========================================="
echo "CXR Classification Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "GPU: $GPU_ID"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Resume: $RESUME"
echo "pos_weight: $POS_WEIGHT"
echo "sqrt_resampling: $SQRT_RESAMPLING"
echo "Start time: $(date)"
echo "=========================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Build command
CMD="python train_config.py --config $CONFIG_FILE"

if [ -n "$CHECKPOINT_DIR" ]; then
    CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
fi

if [ "$RESUME" = "false" ]; then
    CMD="$CMD --no-resume"
fi

# Override config with class imbalance settings
CMD="$CMD --override training.pos_weight=$POS_WEIGHT training.sqrt_resampling=$SQRT_RESAMPLING"

# Run training
echo "Running: $CMD"
$CMD

echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
