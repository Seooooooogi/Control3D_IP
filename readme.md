# Controllable 3D Controllable 3D object Generation with Single Image Prompt

PyTorch Implementation of Controllable 3D Controllable 3D object Generation with Single Image Prompt. Code is built upon [Magic123](https://github.com/guochengqian/Magic123).

# teaser
https://github.com/user-attachments/assets/37ce73c4-0489-4dcd-9044-71ab4847e0bd

# Install

### Install Environment 

```bash
conda create -n ControlIP python=3.10 -y
conda activate ControlIP
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


### Download pre-trained models

* [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) for 3D diffusion prior.
    We use `105000.ckpt` by default, reimplementation borrowed from Stable Diffusion repo, and is available in `guidance/zero123_utils.py`.
    ```bash
    cd pretrained/zero123
    wget https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
    cd ../../
    ```

* [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation.
    We use `dpt_beit_large_512.pt`. Put it in folder `pretrained/midas/`
    ```bash
    mkdir -p pretrained/midas
    cd pretrained/midas
    wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
    cd ../../
    ```

* [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for encoding image prompt.
    We use `ip-adapter_sd15.bin` by default. You can also use plus model. Put it in folder `guidance/adapter/models`
    ```bash 
    cd guidance/adapter/models/
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
    mkdir image_encoder
    cd image_encoder
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin
    cd ../../../../
    ```

# Usage

### Step1: Image Preprocessing 
```
python preprocess_image.py --path /path/to/image 
```

### Step 2: NeRF training 

    ```
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
    ```

### Step 3: DMTet training 
    ```
    CUDA_VISIBLE_DEVICES=0 python main.py -O \
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
    ```
    
### Run on realfusion15 and nerf4 dataset
    ```bash
    bash ours.sh
    ```

### Run on GSO dataset
    ```bash
    bash GSO.sh
    ```

### Run on CO3D dataset
    ```bash
    bash co3d.sh
    ```
    
### Run ablation studies on depth conditioned warmup strategy
    ```bash
    bash anneal_ablation.sh
    ```
