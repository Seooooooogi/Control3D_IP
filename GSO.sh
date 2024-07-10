#!/bin/bash
examples=(
    'shark'
)

for i in "${examples[@]}"; do
    export DATA_DIR='data/GSO'
    export IMAGE_NAME='rgba.png'
    export FILENAME=$(basename $DATA_DIR)
    export dataset=$(basename $(dirname $DATA_DIR))
    CUDA_VISIBLE_DEVICES=1 python main.py -O \
    --text "" \
    --sd_version 1.5 \
    --image ${DATA_DIR}/${i}/${IMAGE_NAME} \
    --workspace out/GSO/${i}_coarse \
    --optim adam \
    --iters 5000 \
    --guidance SD zero123\
    --lambda_guidance 1.0 40\
    --guidance_scale 100 5\
    --latent_iter_ratio 0 \
    --normal_iter_ratio 0.2 \
    --t_range 0.2 0.6 \
    --bg_radius -1 \
    --iters_SD 15 \
    --control \
    --att_scale 0.5 \
    --save_mesh \
    
    CUDA_VISIBLE_DEVICES=1 python main.py -O \
    --text "" \
    --sd_version 1.5 \
    --image ${DATA_DIR}/${i}/${IMAGE_NAME} \
    --workspace out/GSO_fine/${i} \
    --dmtet --init_ckpt out/GSO/${i}_coarse/checkpoints/${i}_coarse.pth \
    --optim adam \
    --iters 5000 \
    --known_view_interval 4 \
    --latent_iter_ratio 0 \
    --guidance SD zero123\
    --lambda_guidance 1e-3 0.01\
    --guidance_scale 100 5\
    --rm_edge \
    --iters_SD 0 \
    --att_scale 0.5 \
    --control \
    --bg_radius -1 \
    --save_mesh \
    
done