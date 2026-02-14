#!/bin/bash
# Batch testing script for CXR classification (calib_distill_kl)
# Tests BioViL, BiomedCLIP, resnet50 × kl_alpha(0.5, 1.0, 2.0) × epoch(5~9)

set -e

##############################################
# CONFIG
##############################################
CONFIG_BASE="/home/yoonji/mrg/cxr_classification/configs"
WEIGHT_BASE="/mnt/HDD/yoonji/mrg/cxr_classification/weight"
OUTPUT_BASE="/home/yoonji/mrg/cxr_classification/test_results"
TEMP_SCALING="false"
GPU_ID=1
##############################################

export CUDA_VISIBLE_DEVICES=$GPU_ID

# 모델별 weight 디렉토리 이름 매핑
declare -A WEIGHT_DIR_MAP
WEIGHT_DIR_MAP["BioViL"]="BioViL_posweight=True_resampling=True_lora_r=16"
WEIGHT_DIR_MAP["BiomedCLIP"]="BiomedCLIP_posweight=True_resampling=True_lora_r=16"
WEIGHT_DIR_MAP["resnet50"]="resnet50_posweight=True_resampling=True"

# 모델별 config 파일 매핑
declare -A CONFIG_MAP
CONFIG_MAP["BioViL"]="${CONFIG_BASE}/biovil_config.yaml"
CONFIG_MAP["BiomedCLIP"]="${CONFIG_BASE}/biomedclip_config.yaml"
CONFIG_MAP["resnet50"]="${CONFIG_BASE}/resnet50_config.yaml"

# resnet50 BioViL BiomedCLIP
for MODEL_NAME in resnet50; do
    for kl_alpha in 0.5 2; do
        for epoch in 9; do

            WEIGHT_DIR="${WEIGHT_DIR_MAP[$MODEL_NAME]}"
            
            # brier #
            # CHECKPOINT_FILE="${WEIGHT_BASE}/${WEIGHT_DIR}/${MODEL_NAME}_brier${kl_alpha}_${epoch}ep.pt"
            # OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_brier/alpha${kl_alpha}_${epoch}ep"

            # focal #
            # CHECKPOINT_FILE="${WEIGHT_BASE}/${WEIGHT_DIR}/${MODEL_NAME}_focal${kl_alpha}_${epoch}ep.pt"
            # OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_focal/alpha${kl_alpha}_${epoch}ep"

            # ours: calib_distill #
            CHECKPOINT_FILE="${WEIGHT_BASE}/${WEIGHT_DIR}/${MODEL_NAME}_calib_distill_latest_kl${kl_alpha}_lr_1_rampup_3_${epoch}ep.pt"
            OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_calib_latest/kl${kl_alpha}_${epoch}ep"

            # 체크포인트 파일 존재 확인
            if [ ! -f "$CHECKPOINT_FILE" ]; then
                echo "[SKIP] Checkpoint not found: $CHECKPOINT_FILE"
                continue
            fi

            echo "=========================================="
            echo "Model: $MODEL_NAME | kl_alpha: $kl_alpha | epoch: $epoch"
            echo "Checkpoint: $CHECKPOINT_FILE"
            echo "Output: $OUTPUT_DIR"
            echo "Start time: $(date)"
            echo "=========================================="

            CONFIG_FILE="${CONFIG_MAP[$MODEL_NAME]}"

            python test_config.py \
                --config "$CONFIG_FILE" \
                --checkpoint "$CHECKPOINT_FILE" \
                --temp_scaling "$TEMP_SCALING" \
                --override "output.dir=$OUTPUT_DIR"

            echo "Done: ${MODEL_NAME} kl${kl_alpha} ${epoch}ep ($(date))"
            echo ""

        done
    done
done

echo "=========================================="
echo "All tests completed! $(date)"
echo "=========================================="
