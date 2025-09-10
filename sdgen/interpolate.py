import torch, time, os
from typing import List
from PIL import Image
from .config import GenerationConfig


def generate_interpolation(cfg: GenerationConfig, pipe, run_dir: str) -> List[str]:
    device = pipe.unet.device
    dtype = next(pipe.unet.parameters()).dtype
    h, w = cfg.height, cfg.width
    latent_h, latent_w = h // 8, w // 8
    in_channels = pipe.unet.config.in_channels if hasattr(pipe.unet, 'config') else 4

    def make_latent(seed: int):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randn((1, in_channels, latent_h, latent_w), generator=g, device=device, dtype=dtype)

    lat_start = make_latent(cfg.interp_seed_start)
    lat_end = make_latent(cfg.interp_seed_end)

    def slerp(t, v0, v1, eps=1e-7):
        v0_u = v0 / (v0.norm(dim=-1, keepdim=True) + eps)
        v1_u = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
        dot = (v0_u * v1_u).sum(-1, keepdim=True).clamp(-1+1e-6, 1-1e-6)
        omega = torch.arccos(dot)
        so = torch.sin(omega)
        return (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1

    v0_flat = lat_start.view(lat_start.size(0), -1)
    v1_flat = lat_end.view(lat_end.size(0), -1)

    paths: List[str] = []
    for idx in range(cfg.interp_frames):
        t = idx / (cfg.interp_frames - 1)
        if cfg.interp_slerp:
            blended_flat = slerp(torch.tensor(t, device=device), v0_flat, v1_flat)
        else:
            blended_flat = (1 - t) * v0_flat + t * v1_flat
        blended = blended_flat.view_as(lat_start)
        t0 = time.time()
        result = pipe(
            cfg.prompt,
            negative_prompt=cfg.negative or None,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            height=h,
            width=w,
            latents=blended.clone()
        )
        img: Image.Image = result.images[0]
        fname = f"{cfg.run_id}-{len(paths)+1:03d}.png"
        fpath = os.path.join(run_dir, fname)
        from . import utils
        utils.save_image_with_meta(img, fpath, {
            'prompt': cfg.prompt,
            'negative': cfg.negative,
            'mode': 'interpolation',
            'model': cfg.model,
            't': f"{t:.4f}",
            'slerp': cfg.interp_slerp,
            'seed_start': cfg.interp_seed_start,
            'seed_end': cfg.interp_seed_end,
            'steps': cfg.steps,
            'guidance': cfg.guidance
        })
        paths.append(fpath)
    return paths
