#!/bin/bash
set -e

exp_name="fm_composed_dress_5e-4"
gpu_id=0
train_gpus="0,1,2,3,4,5,6,7"

log_root="/home/sunyw/CIR/Pic2Word/logs/${exp_name}"
ckpt_dir="${log_root}/checkpoints"

resume_path="/home/sunyw/CIR/Pic2Word/weights/pic2word_model.pt"

loss_type="global"

# -----------------------------
# Global flow config
# 你要的新版默认走：
#   1) 不使用额外 condition
#   2) 起点直接用 image+text 的 composed feature
#   3) composed feature 优先走 pic2word 方式
# 如果训练文本里没有 "*" 占位符，请把 compose_method 改成 add 或 mean。
# -----------------------------
flow_conditioning="disabled"
flow_start_source="composed"
flow_condition_source="image"
flow_compose_method="pic2word"
flow_pic2word_marker="*"
flow_start_text_weight="1.0"
flow_start_image_weight="1.0"
flow_condition_text_weight="1.0"
flow_condition_image_weight="1.0"
flow_path_type="linear"   # linear | geodesic
flow_geodesic_eps="1e-4"
global_start_noise_std="0.0"
disable_delta=0
disable_cond_gate=0

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
    --global-start-noise-std "${global_start_noise_std}"
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
train_dataset_type="flow_matching"
val_dataset_type="fashion-iq"
target_epoch=20

echo "=========================================="
echo "Train to epoch ${target_epoch}"
echo "Resume from: ${resume_path}"
echo "Flow conditioning: ${flow_conditioning}"
echo "Flow start source: ${flow_start_source}"
echo "Flow compose method: ${flow_compose_method}"
echo "Flow path type: ${flow_path_type}"
echo "Val data: ${val_data_path} (${val_dataset_type})"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${train_gpus} python -u src/main_fm.py \
    --save-frequency 20 \
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
    --name "${exp_name}" \
    --flow-num-steps 4 \
    "${extra_flow_args[@]}"
