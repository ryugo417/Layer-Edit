import argparse
import glob
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import AutoencoderKL, FluxPipeline

from FlowEdit_utils import FlowEditFLUX_DUAL_BISTREAM
from rgba_io import preprocess_rgba, preprocess_rgb, latents_norm, latents_denorm, postprocess_rgba


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--exp_yaml", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="/mnt/ssd1/eques/Ledit-bench")
    parser.add_argument("--output_dir", type=str, default="/mnt/ssd1/eques/Ledit_output")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    exp_configs = _load_yaml(args.exp_yaml)

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

    yaml_paths = sorted(glob.glob(os.path.join(args.dataset_dir, "**", "*.yaml"), recursive=True))
    if not yaml_paths:
        raise FileNotFoundError(f"No yaml files found under {args.dataset_dir}")

    for exp_dict in exp_configs:
        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]
        fg_only_start_step = exp_dict.get("fg_only_start_step", None)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for yaml_path in yaml_paths:
            data = _load_yaml(yaml_path)

            src_prompt = data.get("source_prompt", "")
            tar_prompts = _as_list(data.get("target_prompts") or data.get("target_prompt"))
            image_path = data.get("image_path")
            fg_path = data.get("fg_path")
            bg_path = data.get("bg_path")

            if fg_path is None or bg_path is None:
                raise ValueError(f"Missing fg_path/bg_path in {yaml_path}")

            fg_pil = Image.open(fg_path).convert("RGBA")
            bg_pil = Image.open(bg_path).convert("RGB")

            if fg_pil.size != bg_pil.size:
                bg_pil = bg_pil.resize(fg_pil.size, Image.BICUBIC)

            fg = preprocess_rgba(fg_pil, device=device, dtype=weight_dtype, force_divisible_by=16)
            bg = preprocess_rgb(bg_pil, device=device, dtype=weight_dtype, force_divisible_by=16)

            fg01 = (fg * 0.5 + 0.5).clamp(0, 1)
            bg01 = (bg * 0.5 + 0.5).clamp(0, 1)
            a = fg01[:, 3:4]
            scene_rgb01 = fg01[:, :3] * a + bg01 * (1 - a)
            a1 = torch.ones_like(a)
            scene_rgba = torch.cat([scene_rgb01, a1], dim=1) * 2.0 - 1.0

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                z_fg_denorm = pipe.vae.encode(fg).latent_dist.mode()
                z_scene_denorm = pipe.vae.encode(scene_rgba).latent_dist.mode()

            x_fg = latents_norm(pipe, z_fg_denorm)
            x_scene = latents_norm(pipe, z_scene_denorm)

            for tar_num, tar_prompt in enumerate(tar_prompts):
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
                    fg_only_start_step=fg_only_start_step,
                )

                z_fg_out_denorm = latents_denorm(pipe, x_fg_out)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                    fg_rgba_out = pipe.vae.decode(z_fg_out_denorm, return_dict=False)[0]

                fg_imgs = postprocess_rgba(fg_rgba_out)
                fg_out_pil = fg_imgs[0]

                rel = os.path.splitext(os.path.relpath(yaml_path, args.dataset_dir))[0]
                save_dir = os.path.join(args.output_dir, exp_name, rel, f"tar_{tar_num}")
                os.makedirs(save_dir, exist_ok=True)

                fg_out_path = os.path.join(save_dir, "foreground_rgba.png")
                fg_out_pil.save(fg_out_path)

                bg_preview = bg_pil.crop((0, 0, fg_out_pil.size[0], fg_out_pil.size[1]))
                comp = Image.alpha_composite(bg_preview.convert("RGBA"), fg_out_pil.convert("RGBA"))
                comp.save(os.path.join(save_dir, "preview_composite.png"))

                with open(os.path.join(save_dir, "prompts.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tar_prompt}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Image path: {image_path}\n")
                    f.write(f"FG path: {fg_path}\n")
                    f.write(f"BG path: {bg_path}\n")

                print("saved:", fg_out_path)

    print("Done")


if __name__ == "__main__":
    main()
