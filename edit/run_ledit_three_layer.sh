#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/ssd1/eques/DreamLayer/three_layer_flux_lmdb_V1"
DATASET_DIR="${ROOT}/three_layer_ledit_dataset"
OUTPUT_DIR="/mnt/ssd1/eques/Ledit_output_three_layer"

python build_three_layer_ledit_dataset.py \
  --root "${ROOT}" \
  --output-dir "three_layer_ledit_dataset" \
  --mode both \
  --num-targets 8

python run_flowedit_ledit.py \
  --exp_yaml FLUX_exp.yaml \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_model_name_or_path /mnt/ssd1/eques/models/FLUX.1-dev \
  --pretrained_vae_path /mnt/ssd1/eques/models/finetune_VAE \
  --lora_path /mnt/ssd1/eques/models/finetune_VAE/finetune_diffusion
