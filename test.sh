#!/bin/bash
# General testing script for CXR classification
# Edit the CONFIG_FILE and CHECKPOINT_FILE variables below

set -e

##############################################
# EDIT THESE PATHS
##############################################

# for 문
# configs/biovil_config.yaml
# configs/medklip_config.yaml
# configs/densenet121_config.yaml
# configs/resnet50_config.yaml
CONFIG_FILE="configs/biovil_config.yaml"

# for 문
# /mnt/HDD/yoonji/mrg/cxr_classification/weight/BioViL_9ep.pt
# /mnt/HDD/yoonji/mrg/cxr_classification/weight/MedKLIP_9ep.pt
# /mnt/HDD/yoonji/mrg/cxr_classification/weight/densenet121_29ep.pt
# /mnt/HDD/yoonji/mrg/cxr_classification/weight/resnet50_29ep.pt

CHECKPOINT_FILE="/home/yoonji/mrg/cxr_classification/pretrained_weight/biovil_image_model.pt"
# CHECKPOINT_FILE="/mnt/HDD/yoonji/mrg/cxr_classification/weight/BioViL_9ep.pt"

# Temperature scaling: "true" or "false"
# If "true", fits both global and label-wise T on validation set, then tests with all three (uncal + global + label-wise)
# If "false", only uncalibrated results
TEMP_SCALING="false"

# Set GPU (change if needed)
GPU_ID=0
##############################################

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file '$CHECKPOINT_FILE' not found!"
    exit 1
fi

echo "=========================================="
echo "CXR Classification Testing"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Temperature scaling: $TEMP_SCALING"
if [ "$TEMP_SCALING" = "true" ]; then
    echo "  (Will fit global + label-wise T on validation set)"
fi
echo "GPU: $GPU_ID"
echo "Start time: $(date)"
echo "=========================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run testing
python test_config.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_FILE" --temp_scaling "$TEMP_SCALING"

echo "=========================================="
echo "Testing completed!"
echo "End time: $(date)"
echo "=========================================="
