#!/usr/bin/env bash
set -euo pipefail

# Two-stage training:
#   Stage 1: contrastive pretrain qformer for N epochs (flow net frozen)
#   Stage 2: joint train qformer + flow net for M epochs
#
# Usage examples:
#   bash two_stage_qformer_flownet.sh
#   DRY_RUN=1 bash two_stage_qformer_flownet.sh

DRY_RUN="${DRY_RUN:-0}"

exp_name="${EXP_NAME:-qformer10_jointflow10}"
train_gpus="${TRAIN_GPUS:-0}"

# data
train_data_path="${TRAIN_DATA_PATH:-composed_image_retrieval/train.sh}"
val_data_path="${VAL_DATA_PATH:-composed_image_retrieval/val.sh}"
train_dataset_type="${TRAIN_DATASET_TYPE:-cc3m}"
val_dataset_type="${VAL_DATASET_TYPE:-fashion-iq}"

# model/checkpoint
resume_path="${RESUME_PATH:-/home/sunyw/CIR/Pic2Word/weights/pic2word_model.pt}"
model_name="${MODEL_NAME:-ViT-L/14}"

# epoch schedule
stage1_epochs="${STAGE1_EPOCHS:-10}"   # qformer contrastive
stage2_epochs="${STAGE2_EPOCHS:-10}"   # joint qformer + flow net

# optimization
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-16}"
lr="${LR:-5e-5}"
wd="${WD:-0.1}"
warmup="${WARMUP:-500}"
flow_num_steps="${FLOW_NUM_STEPS:-16}"

# loss
loss_type="${LOSS_TYPE:-global}"
lambda_qformer_mod_ret="${LAMBDA_QFORMER_MOD_RET:-1.0}"
lambda_fm="${LAMBDA_FM:-1.0}"
lambda_end="${LAMBDA_END:-0.0}"
lambda_ret="${LAMBDA_RET:-0.0}"

# flow/qformer config
flow_conditioning="${FLOW_CONDITIONING:-enabled}"
flow_start_source="${FLOW_START_SOURCE:-qformer}"
flow_condition_source="${FLOW_CONDITION_SOURCE:-text}"
flow_compose_method="${FLOW_COMPOSE_METHOD:-pic2word}"
flow_pic2word_marker="${FLOW_PIC2WORD_MARKER:-*}"
flow_path_type="${FLOW_PATH_TYPE:-geodesic}"
flow_geodesic_eps="${FLOW_GEODESIC_EPS:-1e-4}"
flow_step_norm_mode="${FLOW_STEP_NORM_MODE:-on}"
flow_step_norm_type="${FLOW_STEP_NORM_TYPE:-expmap}"
flow_hybrid_geodesic_steps="${FLOW_HYBRID_GEODESIC_STEPS:-0}"
global_start_noise_std="${GLOBAL_START_NOISE_STD:-0.0}"
flow_training_objective="${FLOW_TRAINING_OBJECTIVE:-flow_matching}"
flow_block_type="${FLOW_BLOCK_TYPE:-film}"
flow_film_expansion="${FLOW_FILM_EXPANSION:-2}"

disable_delta="${DISABLE_DELTA:-1}"
disable_cond_gate="${DISABLE_COND_GATE:-0}"

qformer_num_layers="${QFORMER_NUM_LAYERS:-2}"
qformer_num_heads="${QFORMER_NUM_HEADS:-8}"
qformer_mlp_ratio="${QFORMER_MLP_RATIO:-4.0}"
qformer_dropout="${QFORMER_DROPOUT:-0.0}"
qformer_query_init_std="${QFORMER_QUERY_INIT_STD:-0.02}"
qformer_use_input_proj="${QFORMER_USE_INPUT_PROJ:-0}"
qformer_image_end_layer="${QFORMER_IMAGE_END_LAYER:--1}"
qformer_text_end_layer="${QFORMER_TEXT_END_LAYER:--1}"

run_cmd() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY_RUN] $*"
    else
        eval "$@"
    fi
}

base_args=(
    --save-frequency 1
    --train-data "${train_data_path}"
    --val-data "${val_data_path}"
    --dataset-type "${train_dataset_type}"
    --dataset-type-val "${val_dataset_type}"
    --warmup "${warmup}"
    --batch-size "${batch_size}"
    --lr "${lr}"
    --wd "${wd}"
    --workers "${num_workers}"
    --loss-type "${loss_type}"
    --openai-pretrained
    --model "${model_name}"
    --flow-num-steps "${flow_num_steps}"
    --global-flow-conditioning "${flow_conditioning}"
    --global-flow-start-source "${flow_start_source}"
    --global-flow-condition-source "${flow_condition_source}"
    --global-flow-compose-method "${flow_compose_method}"
    --global-flow-pic2word-marker "${flow_pic2word_marker}"
    --flow-path-type "${flow_path_type}"
    --flow-geodesic-eps "${flow_geodesic_eps}"
    --flow-step-norm-mode "${flow_step_norm_mode}"
    --flow-step-norm-type "${flow_step_norm_type}"
    --flow-hybrid-geodesic-steps "${flow_hybrid_geodesic_steps}"
    --global-start-noise-std "${global_start_noise_std}"
    --global-flow-block-type "${flow_block_type}"
    --global-flow-film-expansion "${flow_film_expansion}"
    --flow-training-objective "${flow_training_objective}"
    --qformer-num-layers "${qformer_num_layers}"
    --qformer-num-heads "${qformer_num_heads}"
    --qformer-mlp-ratio "${qformer_mlp_ratio}"
    --qformer-dropout "${qformer_dropout}"
    --qformer-query-init-std "${qformer_query_init_std}"
    --qformer-image-end-layer "${qformer_image_end_layer}"
    --qformer-text-end-layer "${qformer_text_end_layer}"
)

if [ "${disable_delta}" = "1" ]; then
    base_args+=(--global-flow-disable-delta)
fi
if [ "${disable_cond_gate}" = "1" ]; then
    base_args+=(--global-flow-disable-cond-gate)
fi
if [ "${qformer_use_input_proj}" = "1" ]; then
    base_args+=(--qformer-use-input-proj)
fi

stage1_name="${exp_name}_stage1_qformer"
stage2_name="${exp_name}_stage2_joint"
stage1_total_epochs="${stage1_epochs}"
stage2_total_epochs=$((stage1_epochs + stage2_epochs))
stage1_ckpt="./logs/${stage1_name}/checkpoints/epoch_${stage1_total_epochs}.pt"

echo "=========================================="
echo "Two-stage training plan"
echo "Stage 1 (qformer contrastive only): ${stage1_epochs} epochs"
echo "Stage 2 (qformer + flow net): ${stage2_epochs} epochs"
echo "Initial resume: ${resume_path}"
echo "Stage1 ckpt : ${stage1_ckpt}"
echo "=========================================="

# Stage 1: freeze flow_net, only optimize qformer contrastive objective
stage1_cmd="CUDA_VISIBLE_DEVICES=${train_gpus} python -u src/main_fm.py ${base_args[*]} \
    --epochs ${stage1_total_epochs} \
    --resume ${resume_path} \
    --name ${stage1_name} \
    --train-qformer \
    --freeze-flow-net \
    --lambda-fm 0.0 \
    --lambda-end 0.0 \
    --lambda-ret 0.0 \
    --lambda-qformer-mod-ret ${lambda_qformer_mod_ret}"
run_cmd "$stage1_cmd"

if [ "$DRY_RUN" != "1" ] && [ ! -f "$stage1_ckpt" ]; then
    echo "[ERROR] Stage1 checkpoint not found: $stage1_ckpt"
    exit 1
fi

# Stage 2: joint train qformer + flow_net from stage1 checkpoint
resume_stage2="$stage1_ckpt"
if [ "$DRY_RUN" = "1" ]; then
    resume_stage2="<stage1_checkpoint>"
fi
stage2_cmd="CUDA_VISIBLE_DEVICES=${train_gpus} python -u src/main_fm.py ${base_args[*]} \
    --epochs ${stage2_total_epochs} \
    --resume ${resume_stage2} \
    --name ${stage2_name} \
    --train-qformer \
    --lambda-fm ${lambda_fm} \
    --lambda-end ${lambda_end} \
    --lambda-ret ${lambda_ret} \
    --lambda-qformer-mod-ret ${lambda_qformer_mod_ret}"
run_cmd "$stage2_cmd"

echo "[DONE] Two-stage training finished."
