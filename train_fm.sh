python -u src/main_fm.py \
    --save-frequency 20 \
    --train-data="composed_image_retrieval/train.sh"  \
    --warmup 50 \
    --batch-size=256 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs=100 \
    --workers=8 \
    --openai-pretrained \
    --model ViT-L/14 \
    --dataset-type flow_matching \
    --resume home/sunyw/CIR/Pic2Word/weights/pic2word_model.pt \
    --name fm_t_v_simple_sphere_w_end
    # optional:
    # --flow-path-type geodesic \
    # --flow-geodesic-eps 1e-4
