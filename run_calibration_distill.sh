#!/bin/bash

# Calibration Distillation 학습 스크립트
# Post-hoc calibration을 training time에 KL divergence로 distillation
#
# 사용법:
#   ./run_calibration_distill.sh

#============================================
# 여기서 직접 수정하세요
#============================================

# 모델 설정
# MODEL="densenet121"
MODEL="resnet50"                  # densenet121, resnet50, BioViL
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
LOAD_EPOCH=4                           # 로드할 체크포인트 epoch
NUM_EPOCHS=10                          # 총 학습 epochs (0~9)
START_EPOCH=5                          # 시작 epoch (LOAD_EPOCH+1)

# Calibration Distillation 설정
WARMUP_EPOCHS=5                        # warmup epochs (epoch 5에서 T fitting 후 바로 distill)
T_UPDATE_FREQ=2                      # T 업데이트 주기 (999=사실상 한 번만)
ALPHA_KL_LIST="0.5 2"           # 실험할 alpha_kl 값들
KL_RAMPUP_EPOCHS=3                     # KL loss ramp-up 기간 (epoch 5,6,7,8,9)
DISTILL_LR_FACTOR=1                  # Distillation 시작 시 LR 감소 비율 ( pretraine model 애들 너무 lora 업데이트 미미해서 수정함)
                    # 저장 파일명 prefix (비어있으면 기본값: {model}_calib_distill_kl{alpha_kl})

#============================================
# 아래는 수정하지 않아도 됩니다
#============================================

# 경로 자동 설정
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

# Alpha KL 값별로 실험
for ALPHA_KL in $ALPHA_KL_LIST; do
    WEIGHT_PREFIX="${MODEL}_calib_distill_latest_kl${ALPHA_KL}_lr_1_rampup_${KL_RAMPUP_EPOCHS}"
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_calib_distill_kl${ALPHA_KL}_lr_1_rampup_${KL_RAMPUP_EPOCHS}"

    echo ""
    echo "========================================"
    echo "Calibration Distillation 학습 (alpha_kl=${ALPHA_KL})"
    echo "========================================"
    echo "모델: $MODEL"
    echo "Config: $CONFIG"
    echo "체크포인트: $LOAD_CHECKPOINT"
    echo "출력: $OUTPUT_DIR"
    echo ""
    echo "시작 epoch: $START_EPOCH"
    echo "종료 epoch: $((NUM_EPOCHS - 1))"
    echo "Warmup epochs: $WARMUP_EPOCHS"
    echo "T 업데이트 주기: ${T_UPDATE_FREQ} epochs"
    echo "Alpha KL: $ALPHA_KL"
    echo "KL ramp-up: ${KL_RAMPUP_EPOCHS} epochs"
    echo "Distill LR factor: $DISTILL_LR_FACTOR"
    echo "pos_weight: $POS_WEIGHT"
    echo "sqrt_resampling: $SQRT_RESAMPLING"
    echo "Weight prefix: ${WEIGHT_PREFIX:-자동 ({model}_calib_distill_kl{alpha_kl})}"
    echo "GPU: $GPU"
    echo "========================================"

    python train_calibration_distill.py \
        --config "$CONFIG" \
        --num_epochs $NUM_EPOCHS \
        --warmup_epochs $WARMUP_EPOCHS \
        --t_update_freq $T_UPDATE_FREQ \
        --alpha_kl $ALPHA_KL \
        --resume "true" \
        --kl_rampup_epochs $KL_RAMPUP_EPOCHS \
        --distill_lr_factor $DISTILL_LR_FACTOR \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --load-checkpoint "$LOAD_CHECKPOINT" \
        --start-epoch $START_EPOCH \
        ${WEIGHT_PREFIX:+--weight-prefix "$WEIGHT_PREFIX"} \
        --override "output.dir=$OUTPUT_DIR" "training.pos_weight=$POS_WEIGHT" "training.sqrt_resampling=$SQRT_RESAMPLING"

    echo ""
    echo "$MODEL alpha_kl=${ALPHA_KL} 학습 완료!"
    echo ""
done

echo "========================================"
echo "모든 alpha_kl 실험 완료!"
echo "========================================"
