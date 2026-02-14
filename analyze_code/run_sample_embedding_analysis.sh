#!/bin/bash
# Test Sample Embedding Visualization
#
# This script extracts feature embeddings from test samples and visualizes
# them using t-SNE with clustering metrics and decision boundaries.

set -e

##############################################
# CONFIGURATION
##############################################

# GPU setting
GPU_ID=0

# Model to analyze (default: resnet50)
MODEL_NAME="${1:-resnet50}"

# Maximum number of samples (reduce for faster computation)
MAX_SAMPLES="${2:-5000}"

# t-SNE perplexity
PERPLEXITY="${3:-30}"

# Model configurations
declare -A CONFIGS
CONFIGS["resnet50"]="configs/resnet50_config.yaml"
CONFIGS["densenet121"]="configs/densenet121_config.yaml"
CONFIGS["biovil"]="configs/biovil_config.yaml"
CONFIGS["medklip"]="configs/medklip_config.yaml"

# Checkpoint paths
declare -A CHECKPOINTS
CHECKPOINTS["resnet50"]="/mnt/HDD/yoonji/mrg/cxr_classification/weight/resnet50_29ep.pt"
CHECKPOINTS["densenet121"]="/mnt/HDD/yoonji/mrg/cxr_classification/weight/densenet121_29ep.pt"
CHECKPOINTS["biovil"]="/mnt/HDD/yoonji/mrg/cxr_classification/weight/BioViL_9ep.pt"
CHECKPOINTS["medklip"]="/mnt/HDD/yoonji/mrg/cxr_classification/weight/MedKLIP_9ep.pt"

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
CHECKPOINT_FILE="${CHECKPOINTS[$MODEL_NAME]}"
OUTPUT_DIR="./outputs/${MODEL_NAME}_experiment/sample_embeddings"

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Test Sample Embedding Visualization"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Output: $OUTPUT_DIR"
echo "Max samples: $MAX_SAMPLES"
echo "t-SNE perplexity: $PERPLEXITY"
echo "GPU: $GPU_ID"
echo "=========================================="

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_FILE"
    exit 1
fi

# Run analysis
echo ""
echo "Running sample embedding analysis..."
python visualize_sample_embeddings.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --perplexity "$PERPLEXITY"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files (2D):"
echo "  - all_labels_single_plot.png: ALL labels in one plot with different colors"
echo "  - all_labels_grid.png: Grid view of all labels"
echo "  - multilabel_overlay.png: Color by number of positive labels"
echo "  - no_finding_vs_abnormal.png: Normal vs abnormal samples"
echo "  - per_label/*.png: Individual label visualizations with decision boundaries"
echo ""
echo "Output files (3D):"
echo "  - 3d/all_labels_single_plot_3d.png: ALL labels in one 3D plot"
echo "  - 3d/all_labels_grid_3d.png: 3D grid view of all labels"
echo "  - 3d/multilabel_overlay_3d.png: 3D multi-label visualization"
echo "  - 3d/no_finding_vs_abnormal_3d.png: 3D normal vs abnormal"
echo "  - 3d/per_label/*.png: 3D individual label visualizations"
echo "  - 3d/interactive_3d_all_labels.html: Interactive 3D with all labels"
echo "  - 3d/interactive_3d_normal_abnormal.html: Interactive 3D normal vs abnormal"
echo ""
echo "Data files:"
echo "  - tsne_embeddings_2d.npy: 2D t-SNE coordinates"
echo "  - tsne_embeddings_3d.npy: 3D t-SNE coordinates"
echo "  - clustering_metrics.csv: Silhouette score, DB index per label"
echo "  - analysis_summary.json: Full analysis summary"
echo "=========================================="
