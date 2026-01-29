import argparse
import hashlib
import io
import json
import os
import random
from pathlib import Path

import lmdb
import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import AutoencoderKL, FluxPipeline

from FlowEdit_utils import FlowEditFLUX_DUAL_BISTREAM
from rgba_io import preprocess_rgba, preprocess_rgb, latents_norm, latents_denorm, postprocess_rgba


LAYER_KEYS = {
    "bg": "layer_0",
    "fg1": "layer_1",
    "fg2": "layer_2",
    "whole": "layer_whole",
}

ADJECTIVES = [
    "red",
    "blue",
    "golden",
    "wooden",
    "metal",
    "transparent",
    "vintage",
    "futuristic",
    "miniature",
    "giant",
]

MATERIALS = [
    "glass",
    "paper",
    "ceramic",
    "plastic",
    "stone",
    "fabric",
]

COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "black",
    "white",
    "silver",
    "gold",
]

SIZES = [
    "tiny",
    "small",
    "large",
    "giant",
    "miniature",
    "oversized",
]

TEXTURES = [
    "smooth",
    "rough",
    "glossy",
    "matte",
    "shiny",
    "rusty",
]

PATTERNS = [
    "striped",
    "polka dot",
    "checkerboard",
    "gradient",
    "camouflage",
]

LIGHTING = [
    "soft lighting",
    "dramatic lighting",
    "neon lighting",
    "warm lighting",
    "cool lighting",
]

CONTENT_OBJECTS = [
    "cat",
    "dog",
    "bird",
    "horse",
    "car",
    "bicycle",
    "tree",
    "flower",
    "chair",
    "lamp",
    "book",
    "mug",
    "backpack",
    "phone",
    "clock",
]

TEMPLATES = [
    "a {adj} {base}",
    "a {base} made of {material}",
    "a {color} {base}",
    "a {size} {base}",
    "a {texture} {base}",
    "a {pattern} {base}",
    "a {base} with {color} accents",
    "a {base} with {texture} surface",
    "a {base} under {lighting}",
]


def parse_group_id(file_name: str) -> str | None:
    marker = "_layer_"
    if marker not in file_name:
        return None
    return file_name.split(marker, 1)[0]


def normalize_base(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "object"
    lowered = cleaned.lower()
    if lowered == "none":
        return "object"
    for prefix in ("a ", "an ", "the "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.strip()
    return cleaned or "object"


def pick_index(seed_key: str, size: int) -> int:
    digest = hashlib.md5(seed_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % size


def build_target_prompt(base: str, seed_key: str, index: int) -> str:
    template = TEMPLATES[pick_index(f"tpl:{seed_key}:{index}", len(TEMPLATES))]
    adj = ADJECTIVES[pick_index(f"adj:{seed_key}:{index}", len(ADJECTIVES))]
    material = MATERIALS[pick_index(f"mat:{seed_key}:{index}", len(MATERIALS))]
    color = COLORS[pick_index(f"color:{seed_key}:{index}", len(COLORS))]
    size = SIZES[pick_index(f"size:{seed_key}:{index}", len(SIZES))]
    texture = TEXTURES[pick_index(f"texture:{seed_key}:{index}", len(TEXTURES))]
    pattern = PATTERNS[pick_index(f"pattern:{seed_key}:{index}", len(PATTERNS))]
    lighting = LIGHTING[pick_index(f"lighting:{seed_key}:{index}", len(LIGHTING))]
    return template.format(
        base=base,
        adj=adj,
        material=material,
        color=color,
        size=size,
        texture=texture,
        pattern=pattern,
        lighting=lighting,
    )


def build_content_swap_prompt(base: str, seed_key: str) -> str:
    base_lower = base.lower()
    choices = [c for c in CONTENT_OBJECTS if c != base_lower]
    if not choices:
        choices = CONTENT_OBJECTS
    choice = choices[pick_index(f"swap:{seed_key}", len(choices))]
    return f"a {choice}"


def build_target_prompts(
    base: str, seed_key: str, num_targets: int, include_swap: bool
) -> list[str]:
    prompts: list[str] = []
    used: set[str] = set()
    if include_swap:
        swap_prompt = build_content_swap_prompt(base, seed_key)
        prompts.append(swap_prompt)
        used.add(swap_prompt)
    idx = 0
    while len(prompts) < num_targets:
        prompt = build_target_prompt(base, seed_key, idx)
        idx += 1
        if prompt in used:
            continue
        prompts.append(prompt)
        used.add(prompt)
        if idx > num_targets * 10:
            break
    return prompts


def load_groups(metadata_path: Path) -> list[dict]:
    groups: dict[str, dict] = {}
    order: list[str] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            file_name = meta.get("file_name", "")
            group_id = parse_group_id(file_name)
            if group_id is None:
                continue
            if group_id not in groups:
                groups[group_id] = {}
                order.append(group_id)
            groups[group_id][file_name] = meta
    return [{"group_id": gid, "items": groups[gid]} for gid in order]


def normalize_layer_tag(layer: str) -> str | None:
    mapping = {
        "layer_1": "layer1",
        "layer_2": "layer2",
        "layer1": "layer1",
        "layer2": "layer2",
    }
    return mapping.get(layer)


def load_target_dataset(path: Path) -> dict:
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not entries:
        return {}
    dataset: dict[str, dict[str, list[dict]]] = {}
    for entry in entries:
        group_id = entry.get("group_id")
        layer = normalize_layer_tag(entry.get("layer", ""))
        if not group_id or layer is None:
            continue
        dataset.setdefault(group_id, {}).setdefault(layer, []).append(entry)
    return dataset


def find_key(items: dict, layer_name: str) -> str | None:
    for key in items.keys():
        if key.endswith(layer_name):
            return key
    return None


def fetch_image(env: lmdb.Environment, key: str) -> Image.Image | None:
    with env.begin() as txn:
        raw = txn.get(key.encode("utf-8"))
    if raw is None:
        return None
    return Image.open(io.BytesIO(raw))


def crop_divisible(image: Image.Image, div: int, mode: str) -> Image.Image:
    image = image.convert(mode)
    w, h = image.size
    w2 = w - (w % div)
    h2 = h - (h % div)
    if (w2, h2) != (w, h):
        image = image.crop((0, 0, w2, h2))
    return image


def composite_scene(fg1: torch.Tensor, fg2: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    fg1_01 = (fg1 * 0.5 + 0.5).clamp(0, 1)
    fg2_01 = (fg2 * 0.5 + 0.5).clamp(0, 1)
    bg_01 = (bg * 0.5 + 0.5).clamp(0, 1)

    a1 = fg1_01[:, 3:4]
    a2 = fg2_01[:, 3:4]
    fg1_rgb = fg1_01[:, :3]
    fg2_rgb = fg2_01[:, :3]

    comp2 = fg2_rgb * a2 + bg_01 * (1 - a2)
    comp1 = fg1_rgb * a1 + comp2 * (1 - a1)
    a1_full = torch.ones_like(a1)
    scene_rgba = torch.cat([comp1, a1_full], dim=1) * 2.0 - 1.0
    return scene_rgba


def composite_preview(
    fg1: Image.Image, fg2: Image.Image, bg: Image.Image, order: str
) -> Image.Image:
    bg_rgba = bg.convert("RGBA")
    if order == "fg1_over_fg2":
        base = Image.alpha_composite(bg_rgba, fg2.convert("RGBA"))
        comp = Image.alpha_composite(base, fg1.convert("RGBA"))
    else:
        base = Image.alpha_composite(bg_rgba, fg1.convert("RGBA"))
        comp = Image.alpha_composite(base, fg2.convert("RGBA"))
    return comp.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--exp_yaml", type=str, required=True)
    parser.add_argument("--root", default="/mnt/ssd1/eques/DreamLayer/three_layer_flux_lmdb_V1")
    parser.add_argument("--lmdb-dir", default="three_layer_flux")
    parser.add_argument("--output_dir", type=str, default="/mnt/ssd1/eques/DreamLayer_output")
    parser.add_argument("--mode", choices=["layer1", "layer2", "both"], default="both")
    parser.add_argument("--num", type=int, default=0, help="Limit number of groups (0 = all)")
    parser.add_argument("--offset", type=int, default=0, help="Start group index in metadata.jsonl")
    parser.add_argument("--num-targets", type=int, default=4)
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument(
        "--target-dataset",
        type=str,
        default="dataset/dataset.yaml",
        help="YAML with group_id/layer/target_prompts (empty to disable)",
    )

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

    base_dir = Path(__file__).resolve().parent
    base_dir = Path(__file__).resolve().parent
    root = Path(args.root)
    metadata_path = root / "metadata.jsonl"
    lmdb_path = root / args.lmdb_dir
    target_dataset_path = Path(args.target_dataset) if args.target_dataset else None
    if target_dataset_path and not target_dataset_path.is_absolute():
        target_dataset_path = base_dir / target_dataset_path

    target_map = {}
    if target_dataset_path:
        if not target_dataset_path.exists():
            print(f"[WARN] Target dataset not found: {target_dataset_path}. Using auto prompts.")
        else:
            target_map = load_target_dataset(target_dataset_path)
    target_dataset_path = Path(args.target_dataset) if args.target_dataset else None
    if target_dataset_path and not target_dataset_path.is_absolute():
        target_dataset_path = base_dir / target_dataset_path

    target_map = {}
    if target_dataset_path:
        if not target_dataset_path.exists():
            raise FileNotFoundError(f"Target dataset not found: {target_dataset_path}")
        target_map = load_target_dataset(target_dataset_path)

    groups = load_groups(metadata_path)
    if args.offset > 0:
        groups = groups[args.offset :]
    if args.num > 0:
        groups = groups[: args.num]
    if not groups:
        raise RuntimeError(f"No groups found in {metadata_path}")

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)

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

        for group in groups:
            group_id = group["group_id"]
            items = group["items"]

            key_bg = find_key(items, LAYER_KEYS["bg"])
            key_fg1 = find_key(items, LAYER_KEYS["fg1"])
            key_fg2 = find_key(items, LAYER_KEYS["fg2"])
            if key_bg is None or key_fg1 is None or key_fg2 is None:
                print(f"[WARN] missing layers for {group_id}")
                continue

            bg_pil = fetch_image(env, key_bg)
            fg1_pil = fetch_image(env, key_fg1)
            fg2_pil = fetch_image(env, key_fg2)
            if bg_pil is None or fg1_pil is None or fg2_pil is None:
                print(f"[WARN] missing image data for {group_id}")
                continue

            bg_pil = crop_divisible(bg_pil, 16, "RGB")
            fg1_pil = crop_divisible(fg1_pil, 16, "RGBA")
            fg2_pil = crop_divisible(fg2_pil, 16, "RGBA")

            bg = preprocess_rgb(bg_pil, device=device, dtype=weight_dtype, force_divisible_by=16)
            fg1 = preprocess_rgba(fg1_pil, device=device, dtype=weight_dtype, force_divisible_by=16)
            fg2 = preprocess_rgba(fg2_pil, device=device, dtype=weight_dtype, force_divisible_by=16)

            scene_rgba = composite_scene(fg1, fg2, bg)

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                z_scene_denorm = pipe.vae.encode(scene_rgba).latent_dist.mode()
            x_scene = latents_norm(pipe, z_scene_denorm)

            text_fg1 = items[key_fg1].get("text", "")
            text_fg2 = items[key_fg2].get("text", "")
            base_fg1 = normalize_base(text_fg1)
            base_fg2 = normalize_base(text_fg2)

            tasks = []
            if args.mode in ("layer1", "both"):
                tasks.append(("layer1", base_fg1, text_fg1, fg1, fg1_pil))
            if args.mode in ("layer2", "both"):
                tasks.append(("layer2", base_fg2, text_fg2, fg2, fg2_pil))

            for layer_tag, base, source_prompt, fg_tensor, fg_pil in tasks:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                    z_fg_denorm = pipe.vae.encode(fg_tensor).latent_dist.mode()
                x_fg = latents_norm(pipe, z_fg_denorm)
                dataset_entries = target_map.get(group_id, {}).get(layer_tag)
                if dataset_entries:
                    prompt_sets = []
                    for entry in dataset_entries:
                        src_prompt = entry.get("source_prompt") or source_prompt or base
                        targets = entry.get("target_prompts") or []
                        if args.num_targets > 0:
                            targets = targets[: args.num_targets]
                        if not targets:
                            continue
                        prompt_sets.append(
                            {
                                "source_prompt": src_prompt,
                                "target_prompts": targets,
                                "edit_type": entry.get("edit_type", ""),
                            }
                        )
                else:
                    targets = build_target_prompts(
                        base=base,
                        seed_key=f"{group_id}:{layer_tag}:{base}",
                        num_targets=args.num_targets,
                        include_swap=not args.no_swap,
                    )
                    prompt_sets = [
                        {"source_prompt": source_prompt or base, "target_prompts": targets, "edit_type": ""}
                    ]

                for prompt_set in prompt_sets:
                    src_prompt = prompt_set["source_prompt"]
                    edit_type = prompt_set.get("edit_type", "")
                    edit_dir = edit_type if edit_type else "auto"
                    save_dir = os.path.join(
                        args.output_dir,
                        exp_name,
                        edit_dir,
                        group_id,
                        layer_tag,
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    bg_dir = os.path.join(save_dir, "background")
                    os.makedirs(bg_dir, exist_ok=True)
                    bg_path = os.path.join(bg_dir, "bg.png")
                    if not os.path.exists(bg_path):
                        bg_pil.convert("RGB").save(bg_path)
                    for tar_num, tar_prompt in enumerate(prompt_set["target_prompts"]):
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
                        z_scene_out_denorm = latents_denorm(pipe, x_scene_out)
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
                            fg_rgba_out = pipe.vae.decode(z_fg_out_denorm, return_dict=False)[0]
                            _ = pipe.vae.decode(z_scene_out_denorm, return_dict=False)[0]

                        fg_out_pil = postprocess_rgba(fg_rgba_out)[0]

                        fg_dir = os.path.join(save_dir, "foreground")
                        comp_dir = os.path.join(save_dir, "preview_composite")
                        os.makedirs(fg_dir, exist_ok=True)
                        os.makedirs(comp_dir, exist_ok=True)

                        fg_out_path = os.path.join(fg_dir, f"tar_{tar_num}.png")
                        fg_out_pil.save(fg_out_path)

                        if layer_tag == "layer1":
                            comp = composite_preview(fg_out_pil, fg2_pil, bg_pil, "fg1_over_fg2")
                        else:
                            comp = composite_preview(fg1_pil, fg_out_pil, bg_pil, "fg1_over_fg2")
                        comp.save(os.path.join(comp_dir, f"tar_{tar_num}.png"))

                        # Save ground-truth originals
                        gt_fg = fg_pil.convert("RGBA")
                        gt_fg.save(os.path.join(fg_dir, "gt.png"))
                        gt_comp = composite_preview(fg1_pil, fg2_pil, bg_pil, "fg1_over_fg2")
                        gt_comp.save(os.path.join(comp_dir, "gt.png"))

                        with open(os.path.join(save_dir, "prompts.txt"), "w", encoding="utf-8") as f:
                            f.write(f"Source prompt: {src_prompt}\n")
                            f.write(f"Target prompt: {tar_prompt}\n")
                            f.write(f"Seed: {seed}\n")
                            f.write(f"Group ID: {group_id}\n")
                            f.write(f"Layer: {layer_tag}\n")
                            if edit_type:
                                f.write(f"Edit type: {edit_type}\n")

                        print("saved:", fg_out_path)

    print("Done")


if __name__ == "__main__":
    main()
