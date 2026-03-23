#!/bin/bash
set -e

exp_name="fm_t_i_mtcir_500k_seq"
gpu_id=0
train_gpus="0,1,2,3,4,5,6,7"

log_root="/home/sunyw/CIR/Pic2Word/logs/${exp_name}"
ckpt_dir="${log_root}/checkpoints"

init_ckpt="/home/sunyw/CIR/Pic2Word/weights/pic2word_model.pt"

loss_type="sequence"

# 每次把总 epoch 设为 20,40,60,80,100
for target_epoch in 5 10
do
    if [ "${target_epoch}" -eq 5 ]; then
        resume_path="${init_ckpt}"
    else
        prev_epoch=$((target_epoch - 5))
        resume_path="${ckpt_dir}/epoch_${prev_epoch}.pt"
    fi

    echo "=========================================="
    echo "Train to epoch ${target_epoch}"
    echo "Resume from: ${resume_path}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${train_gpus} python -u src/main_fm.py \
        --save-frequency 5 \
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
        --name "${exp_name}"

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
            --model ViT-L/14
    done
done