#!/bin/bash

# BCE + Brier Loss 학습 스크립트
# 체크포인트에서 이어서 Brier loss 적용
#
# 사용법:
#   ./run_train_brier.sh

set -e

#============================================
# 여기서 직접 수정하세요
#============================================

# 모델 설정
MODEL="BiomedCLIP"                      # densenet121, resnet50, BioViL, BiomedCLIP
GPU=0

# Class imbalance handling
# True 대문자로 써야함 ㅋ
POS_WEIGHT="True"                      # Use pos_weight in BCEWithLogitsLoss (true/false)
SQRT_RESAMPLING="True"                # Use square root resampling sampler (true/false)

# 경로 설정
WEIGHT_BASE="/mnt/HDD/yoonji/mrg/cxr_classification/weight"
CONFIG_DIR="configs"
OUTPUT_BASE="/home/yoonji/mrg/cxr_classification/outputs"

# 학습 설정
LOAD_EPOCH=4                           # 로드할 체크포인트 epoch
NUM_EPOCHS=10                          # 총 학습 epochs (0~9)
START_EPOCH=5                          # 시작 epoch (LOAD_EPOCH+1)

# Brier Loss 설정
WARMUP_EPOCHS=5                        # warmup epochs (이후부터 Brier loss 시작)
ALPHA_BRIER_LIST="1.0 2.0"        # 실험할 alpha_brier 값들

#============================================
# 아래는 수정하지 않아도 됩니다
#============================================

# 경로 자동 설정 (대소문자 무관하게 파일 찾기)
CONFIG=$(find "$CONFIG_DIR" -maxdepth 1 -iname "${MODEL}_config.yaml" 2>/dev/null | head -1)

# 체크포인트 디렉토리: {MODEL}_posweight={POS_WEIGHT}_resampling={SQRT_RESAMPLING}
if [[ "$MODEL" == "resnet50" ]] || [[ "$MODEL" == "densenet121" ]]; then
    CHECKPOINT_DIR="${WEIGHT_BASE}/${MODEL}_posweight=${POS_WEIGHT}_resampling=${SQRT_RESAMPLING}"
else
    CHECKPOINT_DIR="${WEIGHT_BASE}/${MODEL}_posweight=${POS_WEIGHT}_resampling=${SQRT_RESAMPLING}_lora_r=16"
fi
LOAD_CHECKPOINT="${CHECKPOINT_DIR}/${MODEL}_${LOAD_EPOCH}ep.pt"

# GPU 설정
export CUDA_VISIBLE_DEVICES=$GPU

# 체크포인트 존재 확인
if [[ ! -f "$LOAD_CHECKPOINT" ]]; then
    echo "오류: 체크포인트 없음: $LOAD_CHECKPOINT"
    exit 1
fi

# Config 존재 확인
if [[ ! -f "$CONFIG" ]]; then
    echo "오류: Config 없음: $CONFIG"
    exit 1
fi

# Alpha Brier 값별로 실험
for ALPHA_BRIER in $ALPHA_BRIER_LIST; do
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_brier${ALPHA_BRIER}"

    echo ""
    echo "========================================"
    echo "BCE + Brier Loss 학습 (alpha_brier=${ALPHA_BRIER})"
    echo "========================================"
    echo "모델: $MODEL"
    echo "Config: $CONFIG"
    echo "체크포인트: $LOAD_CHECKPOINT"
    echo "출력: $OUTPUT_DIR"
    echo ""
    echo "시작 epoch: $START_EPOCH"
    echo "종료 epoch: $((NUM_EPOCHS - 1))"
    echo "Warmup epochs: $WARMUP_EPOCHS"
    echo "Alpha Brier: $ALPHA_BRIER"
    echo "pos_weight: $POS_WEIGHT"
    echo "sqrt_resampling: $SQRT_RESAMPLING"
    echo "GPU: $GPU"
    echo "========================================"

    python train_brier.py \
        --config "$CONFIG" \
        --num_epochs $NUM_EPOCHS \
        --warmup_epochs $WARMUP_EPOCHS \
        --alpha_brier $ALPHA_BRIER \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --load-checkpoint "$LOAD_CHECKPOINT" \
        --start-epoch $START_EPOCH \
        --override "output.dir=$OUTPUT_DIR" "training.pos_weight=$POS_WEIGHT" "training.sqrt_resampling=$SQRT_RESAMPLING"

    echo ""
    echo "$MODEL alpha_brier=${ALPHA_BRIER} 학습 완료!"
    echo ""
done

echo "========================================"
echo "모든 alpha_brier 실험 완료!"
echo "========================================"
