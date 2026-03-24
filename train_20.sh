#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 20-epoch Flow Matching training on CC3M_FM
# - Train dataset: CC3M_FM (dataset-type=cc3m)
# - In-training validation: every 10 epochs (implemented in main_fm.py)
# ==========================================================

EXP_NAME="fm_cc3m_20ep"
MODEL_NAME="RN50"
BATCH_SIZE=64
WORKERS=8

# ---------------- Flow matching params (global) ----------------
LOSS_TYPE="global"
FLOW_HIDDEN_DIM=4096
FLOW_TIME_DIM=128
FLOW_NUM_STEPS=4
FLOW_TEMPERATURE=0.07

LAMBDA_FM=1.0
LAMBDA_END=1.0
LAMBDA_RET=0.05
LAMBDA_MID=0.5

GLOBAL_FLOW_CONDITIONING="enabled"
GLOBAL_FLOW_START_SOURCE="text"
GLOBAL_FLOW_CONDITION_SOURCE="image"
GLOBAL_FLOW_COMPOSE_METHOD="add"
GLOBAL_FLOW_START_TEXT_WEIGHT=1.0
GLOBAL_FLOW_START_IMAGE_WEIGHT=1.0
GLOBAL_FLOW_CONDITION_TEXT_WEIGHT=1.0
GLOBAL_FLOW_CONDITION_IMAGE_WEIGHT=1.0
GLOBAL_FLOW_MARKER="*"
GLOBAL_START_NOISE_STD=0.0

python src/main_fm.py \
  --name "${EXP_NAME}" \
  --model "${MODEL_NAME}" \
  --epochs 20 \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --dataset-type cc3m \
  --dataset-type-val cc3m \
  --train-data cc3m_placeholder \
  --lr 5e-5 \
  --wd 0.2 \
  --warmup 1000 \
  --precision amp \
  --loss-type "${LOSS_TYPE}" \
  --flow-hidden-dim "${FLOW_HIDDEN_DIM}" \
  --flow-time-dim "${FLOW_TIME_DIM}" \
  --flow-num-steps "${FLOW_NUM_STEPS}" \
  --flow-temperature "${FLOW_TEMPERATURE}" \
  --lambda-fm "${LAMBDA_FM}" \
  --lambda-end "${LAMBDA_END}" \
  --lambda-ret "${LAMBDA_RET}" \
  --lambda-mid "${LAMBDA_MID}" \
  --global-flow-conditioning "${GLOBAL_FLOW_CONDITIONING}" \
  --global-flow-start-source "${GLOBAL_FLOW_START_SOURCE}" \
  --global-flow-condition-source "${GLOBAL_FLOW_CONDITION_SOURCE}" \
  --global-flow-compose-method "${GLOBAL_FLOW_COMPOSE_METHOD}" \
  --global-flow-start-text-weight "${GLOBAL_FLOW_START_TEXT_WEIGHT}" \
  --global-flow-start-image-weight "${GLOBAL_FLOW_START_IMAGE_WEIGHT}" \
  --global-flow-condition-text-weight "${GLOBAL_FLOW_CONDITION_TEXT_WEIGHT}" \
  --global-flow-condition-image-weight "${GLOBAL_FLOW_CONDITION_IMAGE_WEIGHT}" \
  --global-flow-pic2word-marker "${GLOBAL_FLOW_MARKER}" \
  --global-start-noise-std "${GLOBAL_START_NOISE_STD}" \
  --save-frequency 10 \
  --save-most-recent \
  --report-to tensorboard

# ==========================================================
# Optional: FashionIQ retrieval evaluation after training
# (this is offline test/eval, independent from in-training val-loss)
# ==========================================================
#
# python src/eval_retrieval_fm.py \
#   --name "${EXP_NAME}_fashion_eval" \
#   --model "${MODEL_NAME}" \
#   --dataset-type fashion-iq \
#   --eval-mode fashion \
#   --source-data dress \
#   --batch-size 64 \
#   --workers 8 \
#   --resume "./logs/${EXP_NAME}/checkpoints/epoch_latest.pt"
