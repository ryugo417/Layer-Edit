vae_dir="/mnt/ssd1/eques/models/finetune_VAE"
input_dir="./assets/origin_images"
output_dir="./results/vae_results"

CUDA_VISIBLE_DEVICES=3 python inference/infer_vae.py \
  --pretrained_vae_path ${vae_dir}  \
  --input_dir ${input_dir} \
  --output_dir ${output_dir} \
  --resolution 1024 \
  --dtype bf16
