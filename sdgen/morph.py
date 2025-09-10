import torch, os, time
from typing import List
from PIL import Image
from .config import GenerationConfig
import math, random

def generate_morph(cfg: GenerationConfig, pipe, run_dir: str) -> List[str]:
    """Combined embedding + optional latent interpolation morph.

    Steps:
      1. Encode start/end prompt embeddings.
      2. (Optional) Create start/end latents via morph_seed_start/end.
      3. For each t in frames interpolate embeddings (linear) and latents (linear or slerp) then decode.
    """
    tokenizer = getattr(pipe, 'tokenizer', None)
    text_encoder = getattr(pipe, 'text_encoder', None)
    if tokenizer is None or text_encoder is None:
        raise RuntimeError('Morph not supported for this pipeline')
    device = pipe.unet.device

    def encode(txt: str):
        tokens = tokenizer([txt], padding='max_length', truncation=True, max_length=getattr(tokenizer,'model_max_length',77), return_tensors='pt')
        with torch.no_grad():
            out = text_encoder(tokens.input_ids.to(device))
            if hasattr(out, 'last_hidden_state'):
                emb = out.last_hidden_state
            elif isinstance(out,(list,tuple)):
                emb = out[0]
            else:
                emb = out
        return emb

    emb_start = encode(cfg.morph_from)
    emb_end = encode(cfg.morph_to)
    neg_emb = None
    if cfg.negative:
        try:
            neg_emb = encode(cfg.negative)
        except Exception:
            pass

    # Latent seeds
    latent_start = latent_end = None
    if cfg.morph_latent and cfg.morph_seed_start is not None and cfg.morph_seed_end is not None:
        h, w = cfg.height, cfg.width
        lh, lw = h // 8, w // 8
        in_channels = pipe.unet.config.in_channels if hasattr(pipe.unet,'config') else 4
        dtype = next(pipe.unet.parameters()).dtype
        def make_lat(s):
            g = torch.Generator(device=device).manual_seed(s)
            return torch.randn((1, in_channels, lh, lw), generator=g, device=device, dtype=dtype)
        latent_start = make_lat(cfg.morph_seed_start)
        latent_end = make_lat(cfg.morph_seed_end)

        def slerp_lat(t, v0, v1, eps=1e-7):
            v0_u = v0 / (v0.norm(dim=-1, keepdim=True) + eps)
            v1_u = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
            dot = (v0_u * v1_u).sum(-1, keepdim=True).clamp(-1+1e-6, 1-1e-6)
            omega = torch.arccos(dot)
            so = torch.sin(omega)
            return (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1
    paths: List[str] = []
    def ease(t: float) -> float:
        e = cfg.morph_ease
        if e == 'linear':
            return t
        if e in ('ease','ease-in-out','sine'):
            return 0.5 - 0.5*math.cos(math.pi*t)
        if e == 'ease-in':
            return t*t
        if e == 'ease-out':
            return 1 - (1-t)*(1-t)
        if e == 'quad':
            return t*t
        if e == 'cubic':
            return t*t*t
        return t

    for idx in range(cfg.morph_frames):
        raw_t = idx / (cfg.morph_frames - 1)
        t = ease(raw_t)
        emb_blend = (1 - t) * emb_start + t * emb_end
        latents = None
        if latent_start is not None and latent_end is not None:
            if cfg.morph_slerp:
                flat0 = latent_start.view(latent_start.size(0), -1)
                flat1 = latent_end.view(latent_end.size(0), -1)
                blended_flat = slerp_lat(torch.tensor(t, device=device), flat0, flat1)
                latents = blended_flat.view_as(latent_start)
            else:
                latents = (1 - t) * latent_start + t * latent_end
        # Psychedelic latent noise pulse (adds extra random latent scaled by sinus)
        if latents is not None and cfg.morph_noise_pulse > 0:
            amp = math.sin(raw_t * math.pi)
            if amp > 0:
                noise = torch.randn_like(latents) * (cfg.morph_noise_pulse * amp)
                latents = latents + noise
        kw = dict(
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            height=cfg.height,
            width=cfg.width,
            prompt_embeds=emb_blend
        )
        if latents is not None:
            kw['latents'] = latents.clone()
        if neg_emb is not None and cfg.guidance and cfg.guidance > 1.0:
            kw['negative_prompt_embeds'] = neg_emb
        result = pipe(**kw)
        img: Image.Image = result.images[0]
        # Psychedelic pixel post effects
        if cfg.morph_color_shift or cfg.morph_frame_perturb > 0:
            import numpy as np
            arr = np.array(img).astype('float32')
            if cfg.morph_color_shift:
                # cyclic hue-ish rotation via channel mixing matrix
                ci = cfg.morph_color_intensity
                # simple rotation between channels using a permutation blend
                r,g,b = arr[...,0], arr[...,1], arr[...,2]
                arr[...,0] = (1-ci)*r + ci*g
                arr[...,1] = (1-ci)*g + ci*b
                arr[...,2] = (1-ci)*b + ci*r
            if cfg.morph_frame_perturb > 0:
                # add mild sinusoidal warp in x-direction
                h,w,_ = arr.shape
                strength = cfg.morph_frame_perturb
                yy,xx = np.mgrid[0:h,0:w]
                shift = (np.sin(yy/10 + raw_t*math.pi*2) * strength * 5)
                # apply horizontal shift (nearest)
                xx_shifted = (xx + shift).clip(0,w-1).astype('int')
                arr = arr[np.arange(h)[:,None], xx_shifted]
            arr = arr.clip(0,255).astype('uint8')
            img = Image.fromarray(arr)
        fname = f"{cfg.run_id}-{len(paths)+1:03d}.png"
        fpath = os.path.join(run_dir, fname)
        img.save(fpath)
        paths.append(fpath)
    return paths
