#!/bin/bash

# Calibration Distillation (Jensen-Shannon) 학습 스크립트
# Post-hoc calibration을 training time에 JSD로 distillation
#
# 사용법:
#   ./run_calibration_distill_js.sh

#============================================
# 여기서 직접 수정하세요
#============================================

# 모델 설정
# MODEL="densenet121"
MODEL="densenet121"                  # densenet121, resnet50, BioViL
GPU=1

# 경로 설정
CHECKPOINT_DIR="/mnt/HDD/yoonji/mrg/cxr_classification/weight"
CONFIG_DIR="configs"
OUTPUT_BASE="/home/yoonji/mrg/cxr_classification/outputs"

# 학습 설정
NUM_EPOCHS=30                          # 총 학습 epochs (0~29)
START_EPOCH=15                         # 시작 epoch (14ep 체크포인트 이후)

# Calibration Distillation (JSD) 설정
WARMUP_EPOCHS=15                       # warmup epochs (15부터 distillation 시작)
T_UPDATE_FREQ=5                        # Temperature 업데이트 주기 (epochs)
ALPHA_JSD=1.0                          # JSD loss 최종 weight
JSD_RAMPUP_EPOCHS=5                    # JSD loss ramp-up 기간 (epochs)
DISTILL_LR_FACTOR=0.1                  # Distillation 시작 시 LR 감소 비율
POS_WEIGHT="true"                      # pos_weight 적용 여부 (true/false)

#============================================
# 아래는 수정하지 않아도 됩니다
#============================================

# 경로 자동 설정 (대소문자 무관하게 파일 찾기)
CONFIG=$(find "$CONFIG_DIR" -maxdepth 1 -iname "${MODEL}_config.yaml" 2>/dev/null | head -1)
LOAD_CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -iname "${MODEL}_14ep.pt" 2>/dev/null | head -1)
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_calib_distill_js"

# GPU 설정
export CUDA_VISIBLE_DEVICES=$GPU

# 설정 출력
echo "========================================"
echo "Calibration Distillation (JSD) 학습"
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
echo "Alpha JSD: $ALPHA_JSD"
echo "JSD ramp-up: ${JSD_RAMPUP_EPOCHS} epochs"
echo "Distill LR factor: $DISTILL_LR_FACTOR"
echo "Pos weight: $POS_WEIGHT"
echo "GPU: $GPU"
echo "========================================"

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

# 학습 실행
python train_calibration_distill_js.py \
    --config "$CONFIG" \
    --num_epochs $NUM_EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --t_update_freq $T_UPDATE_FREQ \
    --alpha_jsd $ALPHA_JSD \
    --jsd_rampup_epochs $JSD_RAMPUP_EPOCHS \
    --distill_lr_factor $DISTILL_LR_FACTOR \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --load-checkpoint "$LOAD_CHECKPOINT" \
    --start-epoch $START_EPOCH \
    --pos_weight $POS_WEIGHT \
    --override "output.dir=$OUTPUT_DIR"

echo ""
echo "$MODEL 학습 완료!"
