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

if [ "${disable_delta}" -eq 1 ]; then
    extra_flow_args+=(--global-flow-disable-delta)
fi

if [ "${disable_cond_gate}" -eq 1 ]; then
    extra_flow_args+=(--global-flow-disable-cond-gate)
fi

# 仅新增：validation dataset 设置（配合 main_fm.py 每10 epoch自动跑 val loss）
train_data_path="composed_image_retrieval/train.sh"
val_data_path="composed_image_retrieval/val.sh"
train_dataset_type="cc3m"
val_dataset_type="cc3m"

# 每次把总 epoch 设为 20,40,60,80,100
for target_epoch in 1
do
    if [ "${target_epoch}" -eq 1 ]; then
        resume_path="${init_ckpt}"
    else
        prev_epoch=$((target_epoch - 10))
        resume_path="${ckpt_dir}/epoch_${prev_epoch}.pt"
    fi

    echo "=========================================="
    echo "Train to epoch ${target_epoch}"
    echo "Resume from: ${resume_path}"
    echo "Flow conditioning: ${flow_conditioning}"
    echo "Flow start source: ${flow_start_source}"
    echo "Flow compose method: ${flow_compose_method}"
    echo "Val data: ${val_data_path} (${val_dataset_type})"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${train_gpus} python -u src/main_fm.py \
        --save-frequency 1 \
        --train-data "${train_data_path}" \
        --val-data "${val_data_path}" \
        --dataset-type "${train_dataset_type}" \
        --dataset-type-val "${val_dataset_type}" \
        --warmup 500 \
        --batch-size 256 \
        --lr 1e-5 \
        --wd 0.1 \
        --epochs ${target_epoch} \
        --workers 8 \
        --loss-type ${loss_type} \
        --openai-pretrained \
        --model ViT-L/14 \
        --resume "${resume_path}" \
        --name "${exp_name}" \
        "${extra_flow_args[@]}"

    for cloth_type in dress
    do
        echo "------------------------------------------"
        echo "Eval epoch ${target_epoch}, source=${cloth_type}"
        echo "------------------------------------------"

        CUDA_VISIBLE_DEVICES=${train_gpus} python src/eval_retrieval_fm.py \
            --openai-pretrained \
            --resume "${ckpt_dir}/epoch_${target_epoch}.pt" \
            --eval-mode fashion \
            --loss-type ${loss_type} \
            --source-data "${cloth_type}" \
            --gpu "${gpu_id}" \
            --model ViT-L/14 \
            "${extra_flow_args[@]}"
    done
done

