gpu_id=0
for cloth_type in dress
do
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/eval_retrieval_fm.py \
    --openai-pretrained \
    --resume /home/sunyw/CIR/Pic2Word/logs/fm_t_i_cc3m/checkpoints/epoch_20.pt \
    --eval-mode fashion \
    --source $cloth_type \
    --gpu $gpu_id \
    --model ViT-L/14
done