#!/bin/bash
# Train with frequency-aware Brier calibration loss
#
# Applies different calibration weights based on label frequency:
# - Head group: low weight (already well-calibrated)
# - Medium group: medium weight
# - Tail group: high weight (needs stronger calibration)

set -e

##############################################
# CONFIGURATION
##############################################

# GPU setting
GPU_ID=0

# Model to train (default: resnet50)
MODEL_NAME="${1:-resnet50}"

# Calibration loss settings
WARMUP_EPOCHS="${2:-5}"      # BCE only for first N epochs
ALPHA_HEAD="${3:-0.05}"       # Small weight for head group
ALPHA_MEDIUM="${4:-0.1}"      # Medium weight for medium group
ALPHA_TAIL="${5:-0.2}"        # Large weight for tail group (stronger calibration)

# Model configurations
declare -A CONFIGS
CONFIGS["resnet50"]="configs/resnet50_config.yaml"
CONFIGS["densenet121"]="configs/densenet121_config.yaml"
CONFIGS["biovil"]="configs/biovil_config.yaml"
CONFIGS["medklip"]="configs/medklip_config.yaml"

##############################################
# MAIN
##############################################

# Check if model is valid
if [[ -z "${CONFIGS[$MODEL_NAME]}" ]]; then
    echo "Error: Unknown model '$MODEL_NAME'"
    echo "Available models: ${!CONFIGS[@]}"
    exit 1
fi

CONFIG_FILE="${CONFIGS[$MODEL_NAME]}"

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Frequency-Aware Brier Calibration Training"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_FILE"
echo "GPU: $GPU_ID"
echo ""
echo "Calibration Settings:"
echo "  Warmup epochs (BCE only): $WARMUP_EPOCHS"
echo "  Alpha Head (high-freq): $ALPHA_HEAD"
echo "  Alpha Medium: $ALPHA_MEDIUM"
echo "  Alpha Tail (low-freq): $ALPHA_TAIL"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run training
echo ""
echo "Starting training..."
python train_freq_brier.py \
    --config "$CONFIG_FILE" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --alpha_head "$ALPHA_HEAD" \
    --alpha_medium "$ALPHA_MEDIUM" \
    --alpha_tail "$ALPHA_TAIL" \
    --no-resume

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
