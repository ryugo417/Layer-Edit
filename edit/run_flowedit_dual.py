import torch
from diffusers import FluxPipeline, AutoencoderKL
from PIL import Image
import argparse
import random
import numpy as np
import yaml
import os

from rgba_io import preprocess_rgba, preprocess_rgb, latents_norm, latents_denorm, postprocess_rgba
from FlowEdit_utils import FlowEditFLUX_DUAL_BISTREAM, FlowEditFLUX

def blend_fg_over_bg(fg_rgba_pil: Image.Image, bg_rgb_pil: Image.Image) -> Image.Image:
    fg = np.array(fg_rgba_pil).astype(np.float32) / 255.0  # (H,W,4)
    bg = np.array(bg_rgb_pil.convert("RGB")).astype(np.float32) / 255.0  # (H,W,3)
    a = fg[..., 3:4]
    comp = fg[..., :3] * a + bg * (1 - a)
    comp = (comp * 255).round().clip(0,255).astype(np.uint8)
    return Image.fromarray(comp, mode="RGB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--exp_yaml", type=str, required=True)

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    with open(args.exp_yaml) as f:
        exp_configs = yaml.load(f, Loader=yaml.FullLoader)

    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path, torch_dtype=weight_dtype)
    pipe = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        vae=vae,
    ).to(device)

    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path)
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora()

    scheduler = pipe.scheduler

    for exp_dict in exp_configs:
        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]

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
            fg_path = data_dict["fg_rgba"]
            bg_path = data_dict["bg_rgb"]

            fg_pil = Image.open(fg_path).convert("RGBA")
            bg_pil = Image.open(bg_path).convert("RGB")

            # サイズが違うと座標がズレるので、基本は同サイズを前提
            if fg_pil.size != bg_pil.size:
                # どうしても違う場合は bg を fg に合わせてリサイズ（簡易）
                bg_pil = bg_pil.resize(fg_pil.size, Image.BICUBIC)

            fg = preprocess_rgba(fg_pil, device=device, dtype=weight_dtype, force_divisible_by=16)  # (1,4,H,W) [-1,1]
            bg = preprocess_rgb(bg_pil, device=device, dtype=weight_dtype, force_divisible_by=16)   # (1,3,H,W) [-1,1]

            # scene RGBA = αブレンドした見え（alpha=1）
            fg01 = (fg * 0.5 + 0.5).clamp(0, 1)
            bg01 = (bg * 0.5 + 0.5).clamp(0, 1)
            a = fg01[:, 3:4]
            scene_rgb01 = fg01[:, :3] * a + bg01 * (1 - a)
            a1 = torch.ones_like(a)
            scene_rgba = torch.cat([scene_rgb01, a1], dim=1) * 2.0 - 1.0  # (1,4,H,W) [-1,1]

            # encode fg/scene separately
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                z_fg_denorm = pipe.vae.encode(fg).latent_dist.mode()
                z_scene_denorm = pipe.vae.encode(scene_rgba).latent_dist.mode()

            x_fg = latents_norm(pipe, z_fg_denorm)
            x_scene = latents_norm(pipe, z_scene_denorm)

            for tar_num, tar_prompt in enumerate(tar_prompts):
                # x_fg_out = FlowEditFLUX(
                #     pipe,
                #     scheduler,
                #     x_fg,
                #     src_prompt,
                #     tar_prompt,
                #     negative_prompt="",  # fluxでは未使用
                #     T_steps=T_steps,
                #     n_avg=n_avg,
                #     src_guidance_scale=src_guidance_scale,
                #     tar_guidance_scale=tar_guidance_scale,
                #     n_min=n_min,
                #     n_max=n_max,
                # )
                x_fg_out, x_scene_out = FlowEditFLUX_DUAL_BISTREAM(
                    pipe,
                    scheduler,
                    x_fg=x_fg,
                    x_scene=x_scene,
                    src_prompt=src_prompt,
                    tar_prompt=tar_prompt,
                    T_steps=T_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max,
                )

                # decode layer/scene
                z_fg_out_denorm = latents_denorm(pipe, x_fg_out)
                z_scene_out_denorm = latents_denorm(pipe, x_scene_out)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                    fg_rgba_out = pipe.vae.decode(z_fg_out_denorm, return_dict=False)[0]      # (B,4,H,W) [-1,1]
                    scene_rgba_out = pipe.vae.decode(z_scene_out_denorm, return_dict=False)[0] # (B,4,H,W) [-1,1]

                alpha01 = (fg_rgba_out[:, 3:4] * 0.5 + 0.5).clamp(0, 1)
                rgb01 = (scene_rgba_out[:, :3] * 0.5 + 0.5).clamp(0, 1)
                rgb01 = rgb01 * (alpha01 > 0).to(rgb01.dtype)
                out_rgba = torch.cat([rgb01, alpha01], dim=1) * 2.0 - 1.0

                fg_imgs = postprocess_rgba(out_rgba)
                fg_out_pil = fg_imgs[0]

                save_dir = f"outputs/{exp_name}/FLUX_DUAL/fg_{os.path.basename(fg_path).split('.')[0]}/tar_{tar_num}"
                os.makedirs(save_dir, exist_ok=True)

                fg_out_path = os.path.join(save_dir, "foreground_rgba.png")
                fg_out_pil.save(fg_out_path)

                # preview composite
                bg_preview = bg_pil.crop((0, 0, fg_out_pil.size[0], fg_out_pil.size[1]))
                comp = blend_fg_over_bg(fg_out_pil, bg_preview)
                comp.save(os.path.join(save_dir, "preview_composite.png"))

                with open(os.path.join(save_dir, "prompts.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tar_prompt}\n")
                    f.write(f"Seed: {seed}\n")
                print("saved:", fg_out_path)

    print("Done")

if __name__ == "__main__":
    main()
