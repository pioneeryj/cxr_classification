#!/bin/bash
# Inference on oversampled MIMIC-CXR test samples
# Saves per-sample classification JSON for report generation

set -e

##############################################
# CONFIG - modify these as needed
##############################################
CONFIG_FILE="configs/resnet50_config.yaml"
CHECKPOINT_FILE="/mnt/HDD/yoonji/mrg/cxr_classification/weight/resnet50_posweight=True_resampling=True/resnet50_calib_distill_latest_kl1.0_9ep.pt"
SAMPLE_CSV="/home/yoonji/mrg/cxr_generation/oversampled_subject_10_11_dicom_ids.csv"
OUTPUT_DIR="/home/yoonji/mrg/cxr_classification/inference_results_for_prompt/oversampled_inference"
TEMP_SCALING="false"
GPU_ID=0
##############################################

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Inference on Oversampled Samples"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Sample CSV: $SAMPLE_CSV"
echo "Output: $OUTPUT_DIR"
echo "Temp scaling: $TEMP_SCALING"
echo "Start time: $(date)"
echo "=========================================="

cd /home/yoonji/mrg/cxr_classification

python test_inference.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT_FILE" \
    --sample-csv "$SAMPLE_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --temp_scaling "$TEMP_SCALING"

echo "=========================================="
echo "Inference completed! $(date)"
echo "Results saved to: $OUTPUT_DIR/inference/"
echo "=========================================="
