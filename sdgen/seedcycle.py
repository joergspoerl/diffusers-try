import torch, time, os
from typing import List
from PIL import Image
from .config import GenerationConfig

def generate_seed_cycle(cfg: GenerationConfig, pipe, run_dir: str) -> List[str]:
    device = pipe.unet.device
    dtype = next(pipe.unet.parameters()).dtype
    latent_h, latent_w = cfg.height // 8, cfg.width // 8
    in_channels = getattr(getattr(pipe, 'unet', None), 'config', None).in_channels if hasattr(getattr(pipe, 'unet', None), 'config') else 4
    base_seed = cfg.seed if cfg.seed is not None else int(time.time()) % 2_147_483_647
    base_latent = None
    if cfg.latent_jitter > 0:
        g0 = torch.Generator(device=device).manual_seed(base_seed)
        base_latent = torch.randn((1, in_channels, latent_h, latent_w), generator=g0, device=device, dtype=dtype)
    paths: List[str] = []
    start_time = time.time()
    for i in range(cfg.seed_cycle):
        current_seed = base_seed + i * cfg.seed_step
        gen_i = torch.Generator(device=device).manual_seed(current_seed)
        latents = None
        if base_latent is not None:
            noise = torch.randn_like(base_latent, generator=gen_i) * cfg.latent_jitter
            latents = (base_latent + noise).clone()
        result = pipe(
            cfg.prompt,
            negative_prompt=cfg.negative or None,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            height=cfg.height,
            width=cfg.width,
            generator=gen_i,
            latents=latents
        )
        img: Image.Image = result.images[0]
        fname = f"{cfg.run_id}-{len(paths)+1:03d}.png"
        fpath = os.path.join(run_dir, fname)
        from . import utils
        utils.save_image_with_meta(img, fpath, {
            'prompt': cfg.prompt,
            'negative': cfg.negative,
            'mode': 'seed_cycle',
            'model': cfg.model,
            'current_seed': current_seed,
            'seed_step': cfg.seed_step,
            'latent_jitter': cfg.latent_jitter,
            'steps': cfg.steps,
            'guidance': cfg.guidance
        })
    paths.append(fpath)
    # Progress + ETA
    elapsed = time.time() - start_time
    done = i + 1
    avg = elapsed / done if done else 0.0
    remaining = cfg.seed_cycle - done
    eta = avg * remaining
    total_est = elapsed + eta
    print(f"[seed_cycle {done}/{cfg.seed_cycle}] seed={current_seed} elapsed={elapsed:.1f}s eta={eta:.1f}s total≈{total_est:.1f}s", flush=True)
    return paths
