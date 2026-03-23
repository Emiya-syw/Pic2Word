#!/bin/bash
set -e

exp_name="fm_t_v_mtcir_500k_film_v3"
gpu_id=0
train_gpus="0,1,2,3,4,5,6,7"

log_root="/home/sunyw/CIR/Pic2Word/logs/${exp_name}"
ckpt_dir="${log_root}/checkpoints"

init_ckpt="/home/sunyw/CIR/Pic2Word/weights/pic2word_model.pt"

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
    --global-start-noise-std "${global_start_noise_std}"
)

if [ "${disable_delta}" -eq 1 ]; then
    extra_flow_args+=(--global-flow-disable-delta)
fi

if [ "${disable_cond_gate}" -eq 1 ]; then
    extra_flow_args+=(--global-flow-disable-cond-gate)
fi

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
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${train_gpus} python -u src/main_fm.py \
        --save-frequency 1 \
        --train-data "composed_image_retrieval/train.sh" \
        --warmup 500 \
        --batch-size 256 \
        --lr 1e-5 \
        --wd 0.1 \
        --epochs ${target_epoch} \
        --workers 8 \
        --loss-type ${loss_type} \
        --openai-pretrained \
        --model ViT-L/14 \
        --dataset-type cc3m \
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
            --source "${cloth_type}" \
            --gpu "${gpu_id}" \
            --model ViT-L/14 \
            "${extra_flow_args[@]}"
    done
done
