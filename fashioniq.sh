gpu_id=0
for cloth_type in dress
do
CUDA_VISIBLE_DEVICES=0,1 python src/eval_retrieval_fm.py \
    --openai-pretrained \
    --resume /home/sunyw/CIR/Pic2Word/logs/fm_t_v_mtcir_500k_film/checkpoints/epoch_1.pt \
    --eval-mode fashion \
    --source $cloth_type \
    --gpu $gpu_id \
    --loss-type global \
    --model ViT-L/14
done