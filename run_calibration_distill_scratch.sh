#!/bin/bash

# Calibration Distillation 학습 스크립트 (처음부터 학습)
# checkpoint 로드 없이 epoch 0부터 시작
# epoch 0~(WARMUP-1): BCE only, epoch WARMUP~: BCE + KL ramp-up
#
# 사용법:
#   ./run_calibration_distill_scratch.sh

#============================================
# 여기서 직접 수정하세요
#============================================

# 모델 설정
# MODEL="densenet121"
MODEL="resnet50"                  # resnet50, BioViL, BiomedCLIP
GPU=1

# Class imbalance handling
# True 대문자로 써야함 ㅋ
POS_WEIGHT="True"                      # Use pos_weight in BCEWithLogitsLoss (true/false)
SQRT_RESAMPLING="True"                # Use square root resampling sampler (true/false)
#============================================

# 경로 설정
WEIGHT_BASE="/mnt/HDD/yoonji/mrg/cxr_classification/weight"
CONFIG_DIR="configs"
OUTPUT_BASE="/home/yoonji/mrg/cxr_classification/outputs"

# 학습 설정
NUM_EPOCHS=10                          # 총 학습 epochs (0~9)

# Calibration Distillation 설정
WARMUP_EPOCHS=5                        # warmup epochs (0~4: BCE only, 5~: distill)
T_UPDATE_FREQ=2                        # T 업데이트 주기
ALPHA_KL_LIST="0.5"           # 실험할 alpha_kl 값들
KL_RAMPUP_EPOCHS=3                     # KL loss ramp-up 기간
DISTILL_LR_FACTOR=1                    # Distillation 시작 시 LR 감소 비율
POS_WEIGHT_DECAY="true"               # warmup 이후 pos_weight를 1.0으로 decay (true/false)

#============================================
# 아래는 수정하지 않아도 됩니다
#============================================

# 경로 자동 설정
CONFIG=$(find "$CONFIG_DIR" -maxdepth 1 -iname "${MODEL}_config.yaml" 2>/dev/null | head -1)

# 체크포인트 저장 디렉토리
if [[ "$MODEL" == "resnet50" ]] || [[ "$MODEL" == "densenet121" ]]; then
    CHECKPOINT_DIR="${WEIGHT_BASE}/${MODEL}_posweight=${POS_WEIGHT}_resampling=${SQRT_RESAMPLING}"
else
    CHECKPOINT_DIR="${WEIGHT_BASE}/${MODEL}_posweight=${POS_WEIGHT}_resampling=${SQRT_RESAMPLING}_lora_r=16"
fi

# GPU 설정
export CUDA_VISIBLE_DEVICES=$GPU

# Config 존재 확인
if [[ ! -f "$CONFIG" ]]; then
    echo "오류: Config 없음: $CONFIG"
    exit 1
fi

# Alpha KL 값별로 실험
for ALPHA_KL in $ALPHA_KL_LIST; do
    WEIGHT_PREFIX="${MODEL}_calib_scratch_weight_decay_kl${ALPHA_KL}"
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_calib_distill_scratch_weight_decay_kl${ALPHA_KL}_rampup_${KL_RAMPUP_EPOCHS}"

    echo ""
    echo "========================================"
    echo "Calibration Distillation 처음부터 학습 (alpha_kl=${ALPHA_KL})"
    echo "========================================"
    echo "모델: $MODEL"
    echo "Config: $CONFIG"
    echo "출력: $OUTPUT_DIR"
    echo "Weight prefix: $WEIGHT_PREFIX"
    echo ""
    echo "총 epochs: $NUM_EPOCHS"
    echo "Warmup epochs (BCE only): $WARMUP_EPOCHS"
    echo "T 업데이트 주기: ${T_UPDATE_FREQ} epochs"
    echo "Alpha KL: $ALPHA_KL"
    echo "KL ramp-up: ${KL_RAMPUP_EPOCHS} epochs"
    echo "Distill LR factor: $DISTILL_LR_FACTOR"
    echo "pos_weight: $POS_WEIGHT"
    echo "pos_weight_decay: $POS_WEIGHT_DECAY"
    echo "sqrt_resampling: $SQRT_RESAMPLING"
    echo "GPU: $GPU"
    echo "========================================"

    python train_calibration_distill.py \
        --config "$CONFIG" \
        --num_epochs $NUM_EPOCHS \
        --warmup_epochs $WARMUP_EPOCHS \
        --t_update_freq $T_UPDATE_FREQ \
        --alpha_kl $ALPHA_KL \
        --no-resume \
        --kl_rampup_epochs $KL_RAMPUP_EPOCHS \
        --distill_lr_factor $DISTILL_LR_FACTOR \
        --pos_weight_decay $POS_WEIGHT_DECAY \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --weight-prefix "$WEIGHT_PREFIX" \
        --override "output.dir=$OUTPUT_DIR" "training.pos_weight=$POS_WEIGHT" "training.sqrt_resampling=$SQRT_RESAMPLING"

    echo ""
    echo "$MODEL alpha_kl=${ALPHA_KL} 학습 완료!"
    echo ""
done

echo "========================================"
echo "모든 alpha_kl 실험 완료!"
echo "========================================"
