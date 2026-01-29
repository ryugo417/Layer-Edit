#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=2 python run_flowedit_three_layer.py \
  --exp_yaml FLUX_exp.yaml \
  --root /mnt/ssd1/eques/DreamLayer/three_layer_flux_lmdb_V1 \
  --output_dir /mnt/ssd1/eques/DreamLayer_output \
  --mode both \
  --num-targets 3 \
  --target-dataset dataset/dataset.yaml \
  --pretrained_model_name_or_path /mnt/ssd1/eques/models/FLUX.1-dev \
  --pretrained_vae_path /mnt/ssd1/eques/models/finetune_VAE \
  --lora_path /mnt/ssd1/eques/models/finetune_VAE/finetune_diffusion
