#!/bin/bash
set -e

exp_name="fm_geodesic_text_image_0330"

gpu_id=0
train_gpus="0,1,2,3,4,5,6,7"

log_root="/home/sunyw/CIR/Pic2Word/logs/${exp_name}"
ckpt_dir="${log_root}/checkpoints"

resume_path="/home/sunyw/CIR/Pic2Word/weights/pic2word_model.pt"
# resume_path="/home/sunyw/CIR/Pic2Word/logs/fm_composed_geodesic_image_cond_v_gate_exp_film/checkpoints/epoch_10.pt"
loss_type="global"
lambda_fm="1.0"
lambda_end="0.0"
lambda_ret="0.0002"

# -----------------------------
# Global flow config
# -----------------------------
flow_conditioning="enabled"
flow_start_source="qformer"       # text | image | inversion | composed | qformer
flow_condition_source="text"   # text | image | inversion | composed | qformer
flow_compose_method="pic2word"  # add | mean | pic2word
flow_pic2word_marker="*"
flow_start_text_weight="1.0"
flow_start_image_weight="1.0"
flow_condition_text_weight="1.0"
flow_condition_image_weight="1.0"
flow_geodesic_eps="1e-4"
flow_step_norm_mode="on" # auto: linear->off, geodesic->on
flow_step_norm_type="expmap"   # l2 | expmap
flow_hybrid_geodesic_steps="0" # 0=off; >0 => first s steps geodesic, remaining linear
global_start_noise_std="0.0"
flow_path_type="geodesic"   # linear | geodesic
disable_delta=1
disable_cond_gate=0
flow_training_objective="flow_matching"    # flow_matching | start_end_mse
flow_block_type="${FLOW_BLOCK_TYPE:-film}"      # residual | film
flow_film_expansion="${FLOW_FILM_EXPANSION:-2}"     # used when flow_block_type=film

# -----------------------------
# Single-query Q-Former config
# 仅当 start/condition source 任一为 qformer 时生效
# -----------------------------
training_stage="${TRAINING_STAGE:-3}"  # 1: flow only, 2: qformer only(freeze flow), 3: flow+qformer
train_qformer=1
train_flow_net=1
qformer_num_layers="2"
qformer_num_heads="8"
qformer_mlp_ratio="4.0"
qformer_dropout="0.0"
qformer_query_init_std="0.02"
qformer_use_input_proj=0
qformer_image_end_layer="-1"
qformer_text_end_layer="-1"

if [ "${training_stage}" = "1" ]; then
    train_flow_net=1
    train_qformer=0
elif [ "${training_stage}" = "2" ]; then
    train_flow_net=0
    train_qformer=1
elif [ "${training_stage}" = "3" ]; then
    train_flow_net=1
    train_qformer=1
else
    echo "Unsupported TRAINING_STAGE=${training_stage}, expected 1|2|3"
    exit 1
fi

extra_flow_args=(
    --global-flow-conditioning "${flow_conditioning}"
    --global-flow-start-source "${flow_start_source}"
    --global-flow-condition-source "${flow_condition_source}"
    --global-flow-compose-method "${flow_compose_method}"
    --global-flow-pic2word-marker "${flow_pic2word_marker}"
    --global-flow-start-text-weight "${flow_start_text_weight}"
    --global-flow-start-image-weight "${flow_start_image_weight}"
    --global-flow-condition-text-weight "${flow_condition_text_weight}"
    --global-flow-condition-image-weight "${flow_condition_image_weight}"
    --flow-path-type "${flow_path_type}"
    --flow-geodesic-eps "${flow_geodesic_eps}"
    --flow-step-norm-mode "${flow_step_norm_mode}"
    --flow-step-norm-type "${flow_step_norm_type}"
    --flow-hybrid-geodesic-steps "${flow_hybrid_geodesic_steps}"
    --global-start-noise-std "${global_start_noise_std}"
    --global-flow-block-type "${flow_block_type}"
    --global-flow-film-expansion "${flow_film_expansion}"
    --flow-training-objective "${flow_training_objective}"
)

if [ "${disable_delta}" -eq 1 ]; then
    extra_flow_args+=(--global-flow-disable-delta)
fi

if [ "${disable_cond_gate}" -eq 1 ]; then
    extra_flow_args+=(--global-flow-disable-cond-gate)
fi

if [ "${train_flow_net}" -eq 0 ]; then
    extra_flow_args+=(--freeze-flow-net)
fi

if [ "${flow_start_source}" = "qformer" ] || [ "${flow_condition_source}" = "qformer" ]; then
    extra_flow_args+=(
        --qformer-num-layers "${qformer_num_layers}"
        --qformer-num-heads "${qformer_num_heads}"
        --qformer-mlp-ratio "${qformer_mlp_ratio}"
        --qformer-dropout "${qformer_dropout}"
        --qformer-query-init-std "${qformer_query_init_std}"
        --qformer-image-end-layer "${qformer_image_end_layer}"
        --qformer-text-end-layer "${qformer_text_end_layer}"
    )

    if [ "${train_qformer}" -eq 1 ]; then
        extra_flow_args+=(--train-qformer)
    fi

    if [ "${qformer_use_input_proj}" -eq 1 ]; then
        extra_flow_args+=(--qformer-use-input-proj)
    fi
fi

train_data_path="composed_image_retrieval/train.sh"
val_data_path="composed_image_retrieval/val.sh"
train_dataset_type="cc3m"
val_dataset_type="fashion-iq"
target_epoch=20

echo "=========================================="
echo "Train to epoch ${target_epoch}"
echo "Resume from: ${resume_path}"
echo "Flow conditioning: ${flow_conditioning}"
echo "Flow start source: ${flow_start_source}"
echo "Flow condition source: ${flow_condition_source}"
echo "Training stage: ${training_stage} (train_flow_net=${train_flow_net}, train_qformer=${train_qformer})"
echo "Flow compose method: ${flow_compose_method}"
echo "Flow path type: ${flow_path_type}"
echo "Flow block type: ${flow_block_type} (film_expansion=${flow_film_expansion})"
if [ "${flow_start_source}" = "qformer" ] || [ "${flow_condition_source}" = "qformer" ]; then
    echo "Q-Former: layers=${qformer_num_layers}, heads=${qformer_num_heads}, train_qformer=${train_qformer}"
fi
echo "Loss weights: lambda_fm=${lambda_fm}, lambda_end=${lambda_end}, lambda_ret=${lambda_ret}"
echo "Val data: ${val_data_path} (${val_dataset_type})"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${train_gpus} python -u src/main_fm.py \
    --save-frequency 1 \
    --train-data "${train_data_path}" \
    --val-data "${val_data_path}" \
    --dataset-type "${train_dataset_type}" \
    --dataset-type-val "${val_dataset_type}" \
    --warmup 500 \
    --batch-size 64 \
    --lr 5e-5 \
    --wd 0.1 \
    --epochs ${target_epoch} \
    --workers 16 \
    --loss-type ${loss_type} \
    --openai-pretrained \
    --model ViT-L/14 \
    --resume "${resume_path}" \
    --lambda-fm "${lambda_fm}" \
    --lambda-end "${lambda_end}" \
    --lambda-ret "${lambda_ret}" \
    --name "${exp_name}" \
    --flow-num-steps 16 \
    "${extra_flow_args[@]}"
