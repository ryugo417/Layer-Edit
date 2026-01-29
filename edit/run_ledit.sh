#!/usr/bin/env bash
set -euo pipefail

python run_flowedit_ledit.py \
  --exp_yaml FLUX_exp.yaml \
  --output_dir /mnt/ssd1/eques/Ledit_output \
  --pretrained_model_name_or_path /mnt/ssd1/eques/models/FLUX.1-dev \
  --pretrained_vae_path /mnt/ssd1/eques/models/finetune_VAE \
  --lora_path /mnt/ssd1/eques/models/finetune_VAE/finetune_diffusion
