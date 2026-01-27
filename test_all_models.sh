#!/bin/bash
# Test all models sequentially with temperature scaling

set -e

##############################################
# CONFIGURATION
##############################################

# Temperature scaling: "true" or "false"
TEMP_SCALING="true"

# Set GPU (change if needed)
GPU_ID=0

# Model configurations (config_file checkpoint_file)
declare -a MODELS=(
    "configs/biovil_config.yaml /mnt/HDD/yoonji/mrg/cxr_classification/weight/BioViL_9ep.pt"
    "configs/medklip_config.yaml /mnt/HDD/yoonji/mrg/cxr_classification/weight/MedKLIP_9ep.pt"
    # "configs/densenet121_config.yaml /mnt/HDD/yoonji/mrg/cxr_classification/weight/densenet121_29ep.pt"
    # "configs/resnet50_config.yaml /mnt/HDD/yoonji/mrg/cxr_classification/weight/resnet50_29ep.pt"
)

##############################################

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Testing All Models"
echo "=========================================="
echo "Temperature scaling: $TEMP_SCALING"
echo "GPU: $GPU_ID"
echo "Total models: ${#MODELS[@]}"
echo "Start time: $(date)"
echo "=========================================="

# Loop through all models
for i in "${!MODELS[@]}"; do
    # Parse config and checkpoint
    read -r CONFIG_FILE CHECKPOINT_FILE <<< "${MODELS[$i]}"

    MODEL_NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "Model $MODEL_NUM/${#MODELS[@]}: Testing $(basename $CONFIG_FILE .yaml)"
    echo "=========================================="
    echo "Config: $CONFIG_FILE"
    echo "Checkpoint: $CHECKPOINT_FILE"

    # Check if files exist
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        echo "Skipping this model..."
        continue
    fi

    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "ERROR: Checkpoint file not found: $CHECKPOINT_FILE"
        echo "Skipping this model..."
        continue
    fi

    # Run test
    echo "Running test..."
    python test_config.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_FILE" --temp_scaling "$TEMP_SCALING"

    echo "Model $MODEL_NUM completed!"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All Models Testing Completed!"
echo "End time: $(date)"
echo "=========================================="
