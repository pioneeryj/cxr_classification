#!/bin/bash
# General training script for CXR classification
# Edit the variables below

set -e

##############################################
# EDIT THESE SETTINGS
##############################################

CONFIG_FILE="configs/biovil_config.yaml"
# CONFIG_FILE="configs/medklip_config.yaml"
# CONFIG_FILE="configs/densenet121_config.yaml"
# CONFIG_FILE="configs/resnet50_config.yaml"

# Set GPU (change if needed)
GPU_ID=2

# Checkpoint directory (where to save/load model weights)
CHECKPOINT_DIR="/mnt/HDD/yoonji/mrg/cxr_classification/weight_p13"

# Resume training from checkpoint? (true/false)
RESUME="false"

##############################################

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

echo "=========================================="
echo "CXR Classification Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "GPU: $GPU_ID"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Resume: $RESUME"
echo "Start time: $(date)"
echo "=========================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Build command
CMD="python train_config.py --config $CONFIG_FILE --checkpoint-dir $CHECKPOINT_DIR"

if [ "$RESUME" = "false" ]; then
    CMD="$CMD --no-resume"
fi

# Run training
echo "Running: $CMD"
$CMD

echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
