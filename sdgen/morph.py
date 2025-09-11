import torch, os, time
from typing import List
from PIL import Image
from .config import GenerationConfig
import math, random

def _encode_prompt_embeddings(pipe, tokenizer, text_encoder, texts, device):
    embs = []
    for txt in texts:
        tokens = tokenizer([txt], padding='max_length', truncation=True, max_length=getattr(tokenizer,'model_max_length',77), return_tensors='pt')
        with torch.no_grad():
            out = text_encoder(tokens.input_ids.to(device))
            if hasattr(out, 'last_hidden_state'):
                emb = out.last_hidden_state
            elif isinstance(out,(list,tuple)):
                emb = out[0]
            else:
                emb = out
        embs.append(emb)
    return embs

def _slerp_vec(t, v0, v1, eps=1e-7):
    v0_u = v0 / (v0.norm(dim=-1, keepdim=True) + eps)
    v1_u = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
    dot = (v0_u * v1_u).sum(-1, keepdim=True).clamp(-1+1e-6, 1-1e-6)
    omega = torch.arccos(dot)
    so = torch.sin(omega)
    return (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1

import torch, os, time
from typing import List
from PIL import Image
from .config import GenerationConfig
import math, random

def generate_continuous_morph(cfg: GenerationConfig, pipe, run_dir: str) -> List[str]:
    """Global kontinuierlicher Morph über alle morph_prompts ohne Segment-Reset."""
    from . import modes  # Import hier um zirkuläre Imports zu vermeiden
    
    tokenizer = getattr(pipe, 'tokenizer', None)
    text_encoder = getattr(pipe, 'text_encoder', None)
    if tokenizer is None or text_encoder is None:
        raise RuntimeError('Morph not supported for this pipeline')
    if not cfg.morph_prompts or len(cfg.morph_prompts) < 2:
        return []
    device = pipe.unet.device
    prompts = cfg.morph_prompts
    embeddings = _encode_prompt_embeddings(pipe, tokenizer, text_encoder, prompts, device)
    # Distanzbasierte Segmentierung
    seg_dists = []
    for i in range(len(embeddings)-1):
        d = (embeddings[i]-embeddings[i+1]).pow(2).sum().sqrt().item()
        seg_dists.append(d + 1e-8)
    total = sum(seg_dists)
    cum = [0.0]
    acc = 0.0
    for d in seg_dists:
        acc += d
        cum.append(acc)
    def map_global_t(tg: float):
        target = tg * total
        for si in range(len(seg_dists)):
            if target <= cum[si+1]:
                local = (target - cum[si]) / seg_dists[si]
                return si, min(max(local,0.0),1.0)
        return len(seg_dists)-1, 1.0
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
    total_frames = cfg.morph_frames
    paths: List[str] = []
    prev_img_tensor = None
    import numpy as np
    def effect_weight(raw_t: float) -> float:
        c = cfg.morph_effect_curve
        if c == 'linear': return raw_t
        if c == 'flat': return 1.0
        if c == 'edges': return 1 - 4*(raw_t-0.5)**2
        return math.sin(raw_t * math.pi)
    # optional global latent interpolation
    latent_start = latent_end = None
    if cfg.morph_latent and cfg.morph_seed_start is not None and cfg.morph_seed_end is not None:
        h, w = cfg.height, cfg.width
        lh, lw = h//8, w//8
        in_channels = pipe.unet.config.in_channels if hasattr(pipe.unet,'config') else 4
        dtype = next(pipe.unet.parameters()).dtype
        def make_lat(seed):
            g = torch.Generator(device=device).manual_seed(seed)
            return torch.randn((1,in_channels,lh,lw), generator=g, device=device, dtype=dtype)
        latent_start = make_lat(cfg.morph_seed_start)
        latent_end = make_lat(cfg.morph_seed_end)
    neg_emb = None
    if cfg.negative:
        try:
            neg_emb = _encode_prompt_embeddings(pipe, tokenizer, text_encoder, [cfg.negative], device)[0]
        except Exception:
            pass
    start_time = time.time()
    for i in range(total_frames):
        # Prüfe auf Strg+C Interrupt
        if modes.interrupted:
            print(f"[INFO] Bildgenerierung nach {len(paths)} Frames unterbrochen.")
            break
            
        g_raw = i / (total_frames - 1) if total_frames>1 else 1.0
        g_eased = ease(g_raw)
        seg_idx, local_t = map_global_t(g_eased)
        embA = embeddings[seg_idx]
        embB = embeddings[seg_idx+1]
        
        # Sanftere Interpolation mit mehr Gewichtung
        if cfg.morph_slerp:
            emb = _slerp_vec(torch.tensor(local_t, device=embA.device), embA, embB)
        else:
            # Verwende eine sanftere kubische Interpolation für bessere Übergänge
            smooth_t = local_t * local_t * (3.0 - 2.0 * local_t)  # Smooth step
            emb = (1 - smooth_t) * embA + smooth_t * embB
        latents = None
        if latent_start is not None and latent_end is not None:
            if cfg.morph_slerp:
                flat0 = latent_start.view(latent_start.size(0), -1)
                flat1 = latent_end.view(latent_end.size(0), -1)
                lat_flat = _slerp_vec(torch.tensor(g_eased, device=device), flat0, flat1)
                latents = lat_flat.view_as(latent_start)
            else:
                latents = (1 - g_eased) * latent_start + g_eased * latent_end
            if cfg.morph_noise_pulse > 0:
                amp = math.sin(g_raw * math.pi)
                if amp > 0:
                    latents = latents + torch.randn_like(latents) * (cfg.morph_noise_pulse * amp)
        kw = dict(
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            height=cfg.height,
            width=cfg.width,
            prompt_embeds=emb
        )
        if latents is not None:
            kw['latents'] = latents.clone()
        if neg_emb is not None and cfg.guidance and cfg.guidance > 1.0:
            kw['negative_prompt_embeds'] = neg_emb
        result = pipe(**kw)
        img: Image.Image = result.images[0]
        # Pixel Effekte
        if cfg.morph_color_shift or cfg.morph_frame_perturb > 0:
            arr = np.array(img).astype('float32')
            w_eff = effect_weight(g_raw)
            if cfg.morph_color_shift and w_eff > 0:
                ci = cfg.morph_color_intensity * w_eff
                r,g,b = arr[...,0].copy(), arr[...,1].copy(), arr[...,2].copy()
                arr[...,0] = (1 - ci)*r + ci*g
                arr[...,1] = (1 - ci)*g + ci*b
                arr[...,2] = (1 - ci)*b + ci*r
            if cfg.morph_frame_perturb > 0 and w_eff > 0:
                h,wc,_ = arr.shape
                strength = cfg.morph_frame_perturb * w_eff
                yy,xx = np.mgrid[0:h,0:wc]
                shift = (np.sin(yy/12 + g_raw * math.pi * 2) * strength * 4)
                xx_shifted = (xx + shift).clip(0,wc-1).astype('int')
                arr = arr[np.arange(h)[:,None], xx_shifted]
            arr = arr.clip(0,255).astype('uint8')
            img = Image.fromarray(arr)
        # Temporal Blend
        if cfg.morph_temporal_blend > 0 and prev_img_tensor is not None:
            alpha = cfg.morph_temporal_blend
            cur = np.array(img).astype('float32')
            img = Image.fromarray(((1-alpha)*cur + alpha*prev_img_tensor).clip(0,255).astype('uint8'))
            prev_img_tensor = np.array(img).astype('float32')
        else:
            prev_img_tensor = np.array(img).astype('float32')
        if cfg.morph_smooth:
            try:
                img = img.filter(Image.Filter.SMOOTH)
            except Exception:
                pass
        fname = f"{cfg.run_id}-{len(paths)+1:03d}.png"
        fpath = os.path.join(run_dir, fname)
        from . import utils
        utils.save_image_with_meta(img, fpath, {
            'prompt_sequence': '|'.join(prompts),
            'mode': 'morph_continuous',
            'segment_index': seg_idx,
            'segment_local_t': f"{local_t:.4f}",
            't_global_raw': f"{g_raw:.4f}",
            't_global_eased': f"{g_eased:.4f}",
            'model': cfg.model,
            'steps': cfg.steps,
            'guidance': cfg.guidance,
            'morph_latent': cfg.morph_latent,
            'morph_slerp': cfg.morph_slerp,
            'color_shift': cfg.morph_color_shift,
            'noise_pulse': cfg.morph_noise_pulse,
            'frame_perturb': cfg.morph_frame_perturb,
            'temporal_blend': cfg.morph_temporal_blend,
            'effect_curve': cfg.morph_effect_curve
        })
        paths.append(fpath)
        
        # Update globale Variable für Signal Handler
        modes.current_paths = paths.copy()
        
        # Progress + ETA für jeden Frame
        elapsed = time.time() - start_time
        done = len(paths)
        avg = elapsed / done if done else 0.0
        remaining = total_frames - done
        eta = avg * remaining
        total_est = elapsed + eta
        
        # Debug: Zeige aktuelles Prompt-Segment und Interpolation
        current_prompt_a = prompts[seg_idx][:30] + "..." if len(prompts[seg_idx]) > 30 else prompts[seg_idx]
        current_prompt_b = prompts[seg_idx+1][:30] + "..." if len(prompts[seg_idx+1]) > 30 else prompts[seg_idx+1]
        
        print(f"[morph-cont {done}/{total_frames}] seg {seg_idx+1}/{len(seg_dists)} t={local_t:.3f} elapsed={elapsed:.1f}s eta={eta:.1f}s avg={avg:.1f}s/frame", flush=True)
        print(f"  → {current_prompt_a} → {current_prompt_b}", flush=True)
    return paths

def generate_morph(cfg: GenerationConfig, pipe, run_dir: str) -> List[str]:
    """Wrapper: Wählt continuous oder segmentierten Morph."""
    if cfg.morph_continuous and cfg.morph_prompts and len(cfg.morph_prompts) > 2:
        return generate_continuous_morph(cfg, pipe, run_dir)
    # ursprüngliche segmentierte Implementierung unten
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

    # Determine prompt sequence
    if cfg.morph_prompts and len(cfg.morph_prompts) >= 2:
        prompt_sequence = cfg.morph_prompts
    else:
        prompt_sequence = [cfg.morph_from, cfg.morph_to]
    # Pre-encode all prompt embeddings once
    embeddings = [encode(p) for p in prompt_sequence]
    emb_pairs = list(zip(embeddings[:-1], embeddings[1:]))
    # For legacy metadata
    emb_start = embeddings[0]
    emb_end = embeddings[-1]
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

    prev_img_tensor = None
    import numpy as np
    def effect_weight(raw_t: float) -> float:
        c = cfg.morph_effect_curve
        if c == 'linear':
            return raw_t
        if c == 'flat':
            return 1.0
        if c == 'edges':
            # stronger at ends, weaker mid
            return 1 - 4*(raw_t-0.5)**2
        # default center: strongest middle, smooth edges
        return math.sin(raw_t * math.pi)

    total_frames = cfg.morph_frames
    segments = len(emb_pairs)
    # Distribute frames across segments (at least 2 per segment ideally). We ensure continuity.
    # Strategy: equal proportions; remainder to early segments. Include last frame only in final segment.
    base = max(2, total_frames // segments) if segments>0 else total_frames
    # Build a list with frame counts per segment
    remaining = total_frames
    frames_per_segment = []
    for i in range(segments):
        if i == segments -1:
            cnt = remaining
        else:
            cnt = max(2, remaining // (segments - i))
        frames_per_segment.append(cnt)
        remaining -= cnt
    # Generate
    frame_index = 0
    start_time = time.time()
    for seg_index, ((seg_start, seg_end), seg_frames) in enumerate(zip(emb_pairs, frames_per_segment)):
        for k in range(seg_frames):
            # Avoid duplicating last frame of previous segment: skip first frame if not first segment
            if seg_index > 0 and k == 0:
                continue
            local_raw = k / (seg_frames - 1) if seg_frames > 1 else 1.0
            raw_t = frame_index / (total_frames - 1) if total_frames > 1 else 1.0
            eased_global = ease(raw_t)
            eased_local = ease(local_raw)
            emb_blend = (1 - eased_local) * seg_start + eased_local * seg_end
            latents = None
            if latent_start is not None and latent_end is not None:
                # Latents are still interpolated over full global progress (legacy behaviour)
                g_t = eased_global
                if cfg.morph_slerp:
                    flat0 = latent_start.view(latent_start.size(0), -1)
                    flat1 = latent_end.view(latent_end.size(0), -1)
                    blended_flat = slerp_lat(torch.tensor(g_t, device=device), flat0, flat1)
                    latents = blended_flat.view_as(latent_start)
                else:
                    latents = (1 - g_t) * latent_start + g_t * latent_end
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
            # Pixel post effects with curve shaping
            if cfg.morph_color_shift or cfg.morph_frame_perturb > 0:
                arr = np.array(img).astype('float32')
                w_eff = effect_weight(raw_t)
                if cfg.morph_color_shift and w_eff > 0:
                    ci = cfg.morph_color_intensity * w_eff
                    r, g, b = arr[..., 0].copy(), arr[..., 1].copy(), arr[..., 2].copy()
                    arr[..., 0] = (1 - ci) * r + ci * g
                    arr[..., 1] = (1 - ci) * g + ci * b
                    arr[..., 2] = (1 - ci) * b + ci * r
                if cfg.morph_frame_perturb > 0 and w_eff > 0:
                    h, w, _ = arr.shape
                    strength = cfg.morph_frame_perturb * w_eff
                    yy, xx = np.mgrid[0:h, 0:w]
                    shift = (np.sin(yy / 12 + raw_t * math.pi * 2) * strength * 4)
                    xx_shifted = (xx + shift).clip(0, w - 1).astype('int')
                    arr = arr[np.arange(h)[:, None], xx_shifted]
                arr = arr.clip(0, 255).astype('uint8')
                img = Image.fromarray(arr)
            # Temporal blend for stability
            if cfg.morph_temporal_blend > 0 and prev_img_tensor is not None:
                alpha = cfg.morph_temporal_blend
                arr_cur = np.array(img).astype('float32')
                arr_prev = prev_img_tensor
                arr_blend = (1 - alpha) * arr_cur + alpha * arr_prev
                img = Image.fromarray(arr_blend.clip(0, 255).astype('uint8'))
            # Optional smoothing
            if cfg.morph_smooth:
                try:
                    img = img.filter(Image.Filter.SMOOTH)
                except Exception:
                    pass
            prev_img_tensor = np.array(img).astype('float32')
            fname = f"{cfg.run_id}-{len(paths)+1:03d}.png"
            fpath = os.path.join(run_dir, fname)
            from . import utils
            utils.save_image_with_meta(img, fpath, {
                'prompt_sequence': '|'.join(prompt_sequence),
                'segment_index': seg_index,
                'segment_frames': seg_frames,
                't_segment_raw': f"{local_raw:.4f}",
                't_global_raw': f"{raw_t:.4f}",
                't_global_eased': f"{eased_global:.4f}",
                'mode': 'morph',
                'model': cfg.model,
                'steps': cfg.steps,
                'guidance': cfg.guidance,
                'morph_latent': cfg.morph_latent,
                'morph_slerp': cfg.morph_slerp,
                'color_shift': cfg.morph_color_shift,
                'noise_pulse': cfg.morph_noise_pulse,
                'frame_perturb': cfg.morph_frame_perturb,
                'temporal_blend': cfg.morph_temporal_blend,
                'effect_curve': cfg.morph_effect_curve
            })
            paths.append(fpath)
            frame_index += 1
            # Progress + ETA
            elapsed = time.time() - start_time
            done = frame_index
            avg = elapsed / done if done else 0.0
            remaining = total_frames - done
            eta = avg * remaining
            total_est = elapsed + eta
            print(f"[morph {frame_index}/{total_frames}] segment {seg_index+1}/{segments} elapsed={elapsed:.1f}s eta={eta:.1f}s total≈{total_est:.1f}s", flush=True)
    return paths
