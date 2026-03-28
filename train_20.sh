#!/bin/bash
set -e
# flow_path_type="linear"   # linear | geodesic
for flow_path_type in "geodesic"; do
exp_name="fm_composed_${flow_path_type}_zero_init_cond_v_gate"
gpu_id=0
train_gpus="0,1,2,3,4,5,6,7"

log_root="/home/sunyw/CIR/Pic2Word/logs/${exp_name}"
ckpt_dir="${log_root}/checkpoints"

resume_path="/home/sunyw/CIR/Pic2Word/weights/pic2word_model.pt"

loss_type="global"
lambda_fm="1.0"
lambda_end="0.0"
lambda_ret="0.0002"

# -----------------------------
# Global flow config
# 你要的新版默认走：
#   1) 不使用额外 condition
#   2) 起点直接用 image+text 的 composed feature
#   3) composed feature 优先走 pic2word 方式
# 如果训练文本里没有 "*" 占位符，请把 compose_method 改成 add 或 mean。
# -----------------------------
flow_conditioning="enabled"
flow_start_source="text"
flow_condition_source="inversion"
flow_compose_method="pic2word"
flow_pic2word_marker="*"
flow_start_text_weight="1.0"
flow_start_image_weight="1.0"
flow_condition_text_weight="1.0"
flow_condition_image_weight="1.0"
flow_geodesic_eps="1e-4"
flow_step_norm_mode="on" # auto: linear->off, geodesic->on
flow_step_norm_type="l2"   # l2 | expmap
flow_hybrid_geodesic_steps="0" # 0=off; >0 => first s steps geodesic, remaining linear
global_start_noise_std="0.0"
disable_delta=1
disable_cond_gate=0
flow_block_type="${FLOW_BLOCK_TYPE:-residual}"      # residual | film
flow_film_expansion="${FLOW_FILM_EXPANSION:-2}"     # used when flow_block_type=film

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
)

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
val_dataset_type="fashion-iq"
target_epoch=20

echo "=========================================="
echo "Train to epoch ${target_epoch}"
echo "Resume from: ${resume_path}"
echo "Flow conditioning: ${flow_conditioning}"
echo "Flow start source: ${flow_start_source}"
echo "Flow compose method: ${flow_compose_method}"
echo "Flow path type: ${flow_path_type}"
echo "Flow block type: ${flow_block_type} (film_expansion=${flow_film_expansion})"
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
    --batch-size 256 \
    --lr 5e-5 \
    --wd 0.1 \
    --epochs ${target_epoch} \
    --workers 8 \
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
done
