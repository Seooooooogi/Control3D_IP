#!/bin/bash
examples=(
    'banana'
    'barbie_cake'
    'bird_sparrow'
    'blue_bird'
    'cactus'
    'cat_statue'
    'colorful_teapot'
    'fish_real_nemo'
    'metal_dragon_statue'
    'microphone'
    'stone_dragon_statue'
    'teddy_bear'
    'two_cherries'
    'two_donuts'
    'watercolor_horse'
)

for i in "${examples[@]}"; do
    export DATA_DIR='data/realfusion15'
    export IMAGE_NAME='rgba.png'
    export FILENAME=$(basename $DATA_DIR)
    export dataset=$(basename $(dirname $DATA_DIR))
    CUDA_VISIBLE_DEVICES=0 python main.py -O \
    --text "" \
    --sd_version 1.5 \
    --image "${DATA_DIR}/${i}/${IMAGE_NAME}" \
    --workspace "out/ours/${i}_coarse" \
    --optim adam \
    --iters 5000 \
    --guidance SD zero123 \
    --lambda_guidance 1.0 40\
    --guidance_scale 100 5\
    --latent_iter_ratio 0 \
    --normal_iter_ratio 0.2 \
    --t_range 0.2 0.6 \
    --bg_radius -1 \
    --iters_SD 15 \
    --control \
    --att_scale 1.0 \
    --save_mesh
    
    export DATA_DIR='data/realfusion15'
    export IMAGE_NAME='rgba.png'
    export FILENAME=$(basename $DATA_DIR)
    export dataset=$(basename $(dirname $DATA_DIR))
    CUDA_VISIBLE_DEVICES=0 python main.py -O \
    --text "" \
    --sd_version 1.5 \
    --image "${DATA_DIR}/${i}/${IMAGE_NAME}" \
    --workspace "out/ours_fine/${i}" \
    --dmtet --init_ckpt "out/ours/${i}_coarse/checkpoints/${i}_coarse.pth" \
    --optim adam \
    --iters 5000 \
    --known_view_interval 4 \
    --latent_iter_ratio 0 \
    --guidance SD zero123 \
    --lambda_guidance 1e-3 0.01\
    --guidance_scale 100 5\
    --rm_edge \
    --control \
    --iters_SD 0 \
    --att_scale 1.0 \
    --bg_radius -1 \
    --save_mesh
done

examples=(
    'chair'
    'drums'
    'ficus'
    'mic'
)

for i in "${examples[@]}"; do
    export RUN_ID='baseline_'
    export DATA_DIR='data/nerf4'
    export IMAGE_NAME='rgba.png'
    export FILENAME=$(basename $DATA_DIR)
    export dataset=$(basename $(dirname $DATA_DIR))
    CUDA_VISIBLE_DEVICES=0 python main.py -O \
    --text "" \
    --sd_version 1.5 \
    --image "${DATA_DIR}/${i}/${IMAGE_NAME}" \
    --workspace "out/ours/${i}_coarse" \
    --optim adam \
    --iters 5000 \
    --guidance SD zero123 \
    --lambda_guidance 1.0 40\
    --guidance_scale 100 5\
    --latent_iter_ratio 0 \
    --normal_iter_ratio 0.2 \
    --t_range 0.2 0.6 \
    --iters_SD 15 \
    --att_scale 0.5 \
    --control \
    --bg_radius -1 \
    --save_mesh
    
    export RUN_ID='baseline_'
    export RUN_ID2='fine'
    export DATA_DIR='data/nerf4'
    export IMAGE_NAME='rgba.png'
    export FILENAME=$(basename $DATA_DIR)
    export dataset=$(basename $(dirname $DATA_DIR))
    CUDA_VISIBLE_DEVICES=0 python main.py -O \
    --text ""\
    --sd_version 1.5 \
    --image "${DATA_DIR}/${i}/${IMAGE_NAME}" \
    --workspace "out/ours_fine/${i}" \
    --dmtet --init_ckpt "out/ours/${i}_coarse/checkpoints/${i}_coarse.pth" \
    --optim adam \
    --iters 5000 \
    --known_view_interval 4 \
    --latent_iter_ratio 0 \
    --guidance SD zero123 \
    --lambda_guidance 1e-3 0.01\
    --guidance_scale 100 5\
    --iters_SD 0 \
    --att_scale 0.5 \
    --control \
    --rm_edge \
    --bg_radius -1 \
    --save_mesh
done