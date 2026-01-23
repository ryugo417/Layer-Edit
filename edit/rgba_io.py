import numpy as np
import torch
from PIL import Image

def preprocess_rgb(
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    force_divisible_by: int = 16,
):
    """
    PIL(RGB/RGBA) -> torch (1,3,H,W) in [-1,1]
    """
    image = image.convert("RGB")

    w, h = image.size
    w2 = w - (w % force_divisible_by)
    h2 = h - (h % force_divisible_by)
    if (w2, h2) != (w, h):
        image = image.crop((0, 0, w2, h2))

    arr = np.array(image).astype(np.float32) / 255.0  # [0,1], (H,W,3)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    x = x * 2.0 - 1.0
    return x.to(device=device, dtype=dtype)

def preprocess_rgba(
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    force_divisible_by: int = 16,
    zero_rgb_where_alpha_zero: bool = True,
):
    """
    PIL(RGBA or RGB) -> torch (1,4,H,W) in [-1,1]
    """
    image = image.convert("RGBA")

    # crop to multiple of N (FlowEdit側でもやってるけど、念のため共通化)
    w, h = image.size
    w2 = w - (w % force_divisible_by)
    h2 = h - (h % force_divisible_by)
    if (w2, h2) != (w, h):
        image = image.crop((0, 0, w2, h2))

    arr = np.array(image).astype(np.float32) / 255.0  # [0,1], (H,W,4)
    if zero_rgb_where_alpha_zero:
        a = arr[..., 3:4]
        arr[..., :3] = np.where(a <= 1e-6, 0.0, arr[..., :3])

    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,4,H,W)
    x = x * 2.0 - 1.0  # [-1,1]
    return x.to(device=device, dtype=dtype)

def postprocess_rgba(x: torch.FloatTensor):
    """
    torch (B,4,H,W) in [-1,1] -> list[PIL.Image(RGBA)]
    """
    x = x.detach().float()
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)  # [0,1]
    imgs = []
    for i in range(x.shape[0]):
        arr = (x[i].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)  # (H,W,4)
        imgs.append(Image.fromarray(arr, mode="RGBA"))
    return imgs

def latents_norm(pipe, z):
    # FLUX系は shift_factor があることが多い。無い場合は 0 として扱う
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
    return (z - shift) * scale

def latents_denorm(pipe, z):
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
    return (z / scale) + shift
