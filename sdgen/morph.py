import torch, os, time
from typing import List
from PIL import Image
from .config import GenerationConfig

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
    for idx in range(cfg.morph_frames):
        t = idx / (cfg.morph_frames - 1)
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
        fname = f"{cfg.run_id}-{len(paths)+1:03d}.png"
        fpath = os.path.join(run_dir, fname)
        img.save(fpath)
        paths.append(fpath)
    return paths
