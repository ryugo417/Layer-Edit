CUDA_VISIBLE_DEVICES=3 python run_flowedit_dual.py \
  --exp_yaml /home/eques/morita/AlphaVAE/edit/FLUX_exp.yaml \
  --pretrained_model_name_or_path /mnt/ssd1/eques/models/FLUX.1-dev \
  --pretrained_vae_path /mnt/ssd1/eques/models/finetune_VAE \
  --lora_path /mnt/ssd1/eques/models/finetune_VAE/finetune_diffusion \
  --dtype fp16 