import torch
from diffusers import FluxPipeline, AutoencoderKL
from PIL import Image
import argparse
import random
import numpy as np
import yaml
import os

from FlowEdit_utils import FlowEditFLUX
from rgba_io import preprocess_rgba, postprocess_rgba, latents_norm, latents_denorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--exp_yaml", type=str, default="/home/eques/morita/AlphaVAE/edit/FLUX_exp.yaml")

    # 追加：AlphaVAE VAE + LoRA を指定できるようにする
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/home/eques/morita/AlphaVAE/models/FLUX.1-dev")
    parser.add_argument("--pretrained_vae_path", type=str, default="/mnt/ssd1/eques/models/finetune_VAE", help="AlphaVAEのRGBA VAEのパス/ID")
    parser.add_argument("--lora_path", type=str, default="/mnt/ssd1/eques/models/finetune_VAE/finetune_diffusion", help="AlphaVAEのdiffusion LoRA（任意）")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # exp yaml
    with open(args.exp_yaml) as f:
        exp_configs = yaml.load(f, Loader=yaml.FullLoader)

    # ---- Load RGBA VAE + FLUX ----
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path, torch_dtype=weight_dtype)

    pipe = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        vae=vae,
    ).to(device)

    # 任意：AlphaVAE diffusion LoRA を適用（強く推奨）
    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path)
        # diffusersのバージョンによっては fuse_lora できる
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora()

    scheduler = pipe.scheduler

    # ---- Run experiments ----
    for exp_dict in exp_configs:
        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]

        # seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dataset_yaml = exp_dict["dataset_yaml"]
        with open(dataset_yaml) as f:
            dataset_configs = yaml.load(f, Loader=yaml.FullLoader)

        for data_dict in dataset_configs:
            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            image_src_path = data_dict["input_img"]

            # ---- RGBA preprocess ----
            image = Image.open(image_src_path)
            image_src = preprocess_rgba(image, device=device, dtype=weight_dtype, force_divisible_by=16)

            # ---- Encode to latents ----
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                z_denorm = pipe.vae.encode(image_src).latent_dist.mode()

            x0_src = latents_norm(pipe, z_denorm).to(device)

            for tar_num, tar_prompt in enumerate(tar_prompts):
                x0_tar = FlowEditFLUX(
                    pipe,
                    scheduler,
                    x0_src,
                    src_prompt,
                    tar_prompt,
                    negative_prompt="",  # fluxでは未使用
                    T_steps=T_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max,
                )

                # ---- Decode ----
                z_tar_denorm = latents_denorm(pipe, x0_tar)

                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                    rgba = pipe.vae.decode(z_tar_denorm, return_dict=False)[0]  # (B,4,H,W)

                images = postprocess_rgba(rgba)  # list[PIL RGBA]

                src_prompt_txt = os.path.basename(image_src_path).split(".")[0]
                tar_prompt_txt = str(tar_num)

                save_dir = f"outputs/{exp_name}/FLUX_RGBA/src_{src_prompt_txt}/tar_{tar_prompt_txt}"
                os.makedirs(save_dir, exist_ok=True)

                out_path = (
                    f"{save_dir}/output_T{T_steps}_avg{n_avg}"
                    f"_cfgS{src_guidance_scale}_cfgT{tar_guidance_scale}"
                    f"_nmin{n_min}_nmax{n_max}_seed{seed}.png"
                )
                images[0].save(out_path)
                print("saved:", out_path)

                with open(f"{save_dir}/prompts.txt", "w", encoding="utf-8") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tar_prompt}\n")
                    f.write(f"Seed: {seed}\n")

    print("Done")


if __name__ == "__main__":
    main()
