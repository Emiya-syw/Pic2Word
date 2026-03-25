gpu_id=0

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
flow_step_norm_mode="auto" # auto: linear->off, geodesic->on
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
    --flow-step-norm-mode "${flow_step_norm_mode}"
    --global-start-noise-std "${global_start_noise_std}"
)

for cloth_type in dress
do
CUDA_VISIBLE_DEVICES=0,1 python src/eval_retrieval_fm.py \
    --openai-pretrained \
    --resume /home/sunyw/CIR/Pic2Word/logs/fm_composed_dress/checkpoints/epoch_1.pt \
    --eval-mode fashion \
    --source $cloth_type \
    --gpu $gpu_id \
    --loss-type global \
    --model ViT-L/14 \
    "${extra_flow_args[@]}"
done
