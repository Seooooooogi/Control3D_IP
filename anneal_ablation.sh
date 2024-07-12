                                                                                                                                                                                                                                                                                        #!/bin/bash
warmups=(
    0
    15
    30
    45
)
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
    for j in "${warmups[@]}"; do
        export RUN_ID='baseline_'
        export DATA_DIR='data/realfusion15'
        export IMAGE_NAME='rgba.png'
        export FILENAME=$(basename $DATA_DIR)
        export dataset=$(basename $(dirname $DATA_DIR))
        CUDA_VISIBLE_DEVICES=1 python main.py -O \
        --text "" \
        --sd_version 1.5 \
        --image ${DATA_DIR}/${i}/${IMAGE_NAME} \
        --workspace out/depth_anneal_ablation/${j}_${i}_coarse \
        --optim adam \
        --iters 5000 \
        --guidance SD zero123\
        --lambda_guidance 1.0 40\
        --guidance_scale 100 5\
        --latent_iter_ratio 0 \
        --normal_iter_ratio 0.2 \
        --t_range 0.2 0.6 \
        --bg_radius -1 \
        --control \
        --att_scale 0.5 \
        --iters_SD ${j} \
        --save_mesh
    
        export DATA_DIR='data/realfusion15'
        export IMAGE_NAME='rgba.png'
        export FILENAME=$(basename $DATA_DIR)
        export dataset=$(basename $(dirname $DATA_DIR))
        CUDA_VISIBLE_DEVICES=1 python main.py -O \
        --text "" \
        --sd_version 1.5 \
        --image ${DATA_DIR}/${i}/${IMAGE_NAME} \
        --workspace out/depth_anneal_ablation/${j}_${i}_fine \
        --dmtet --init_ckpt out/depth_anneal_ablation/${j}_${i}_coarse/checkpoints/${j}_${i}_coarse.pth \
        --optim adam \
        --iters 5000 \
        --known_view_interval 4 \
        --latent_iter_ratio 0 \
        --guidance SD zero123\
        --lambda_guidance 1e-3 0.01\
        --guidance_scale 100 5\
        --control \
        --rm_edge \
        --iters_SD 0 \
        --bg_radius -1 \
        --att_scale 0.5 \
        --save_mesh 
    done
done