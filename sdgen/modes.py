from typing import List
import os, time, torch
from PIL import Image
from .config import GenerationConfig
from .pipeline import infer_defaults
from . import utils
from . import interpolate, seedcycle, morph, video

# Placeholder: This module would be filled by migrating logic stepwise from generate.py

def run_generation(cfg: GenerationConfig, pipe) -> List[str]:
    steps, guidance = infer_defaults(cfg.model, cfg.steps, cfg.guidance)
    # Prepare run directory
    if cfg.make_run_dir:
        # Morph spezifischer run_id Name falls zutreffend
        if cfg.morph_from and cfg.morph_to and cfg.morph_frames > 1:
            run_id = cfg.run_id or utils.build_morph_run_id(cfg.morph_from, cfg.morph_to)
        else:
            run_id = cfg.run_id or utils.build_run_id(cfg.prompt)
        run_dir = cfg.run_dir or os.path.join(cfg.outdir, run_id)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_id = cfg.run_id or 'run'
        run_dir = cfg.run_dir or cfg.outdir
        os.makedirs(run_dir, exist_ok=True)
    cfg.run_id = run_id
    cfg.run_dir = run_dir
    # Decide mode
    paths: List[str] = []
    # Fill defaults
    cfg.steps, cfg.guidance = infer_defaults(cfg.model, cfg.steps, cfg.guidance)

    # Interpolation priority after morph
    if cfg.morph_from and cfg.morph_to and cfg.morph_frames > 1:
        cfg.mode = 'morph'
    elif cfg.interp_seed_start is not None and cfg.interp_seed_end is not None and cfg.interp_frames > 1:
        cfg.mode = 'interpolation'
    elif cfg.seed_cycle > 0:
        cfg.mode = 'seed_cycle'
    elif cfg.images > 1:
        cfg.mode = 'batch'
    else:
        cfg.mode = 'single'

    # Generate
    if cfg.info_only:
        # Simple model stats output and return empty
        stats = {}
        for comp in ['unet','vae','text_encoder','text_encoder_2']:
            m = getattr(pipe, comp, None)
            if m is not None:
                try:
                    stats[comp] = sum(p.numel() for p in m.parameters())
                except Exception:
                    pass
        stats['total'] = sum(stats.values())
        print('Model Stats:')
        for k,v in stats.items():
            print(f" - {k}: {v}")
        return []

    # Enable / disable attention slicing
    if not cfg.no_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    # Offload
    if cfg.seq_offload:
        try:
            pipe.enable_sequential_cpu_offload()
            print('Sequentielles CPU Offload aktiv')
        except Exception as e:
            print('Warnung seq_offload:', e)
    elif cfg.cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            print('CPU Offload aktiv')
        except Exception as e:
            print('Warnung cpu_offload:', e)

    if cfg.mode == 'morph':
        paths = morph.generate_morph(cfg, pipe, run_dir)
    elif cfg.mode == 'interpolation':
        paths = interpolate.generate_interpolation(cfg, pipe, run_dir)
    elif cfg.mode == 'seed_cycle':
        paths = seedcycle.generate_seed_cycle(cfg, pipe, run_dir)
    else:
        # single or batch
        g = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
        if cfg.seed is not None:
            g = g.manual_seed(cfg.seed)
        if cfg.mode == 'batch':
            result = pipe(
                cfg.prompt,
                negative_prompt=cfg.negative or None,
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance,
                height=cfg.height,
                width=cfg.width,
                num_images_per_prompt=cfg.images,
                generator=g
            )
            for i, img in enumerate(result.images, 1):
                fname = f"{cfg.run_id}-{i:03d}.png"
                fpath = os.path.join(run_dir, fname)
                utils.save_image_with_meta(img, fpath, {
                    'prompt': cfg.prompt,
                    'negative': cfg.negative,
                    'model': cfg.model,
                    'mode': cfg.mode,
                    'index': i,
                    'seed': cfg.seed,
                    'steps': cfg.steps,
                    'guidance': cfg.guidance
                })
                paths.append(fpath)
        else:
            result = pipe(
                cfg.prompt,
                negative_prompt=cfg.negative or None,
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance,
                height=cfg.height,
                width=cfg.width,
                generator=g
            )
            img: Image.Image = result.images[0]
            fname = f"{cfg.run_id}-001.png"
            fpath = os.path.join(run_dir, fname)
            utils.save_image_with_meta(img, fpath, {
                'prompt': cfg.prompt,
                'negative': cfg.negative,
                'model': cfg.model,
                'mode': cfg.mode,
                'seed': cfg.seed,
                'steps': cfg.steps,
                'guidance': cfg.guidance
            })
            paths.append(fpath)
    # NOTE: (Bugfix) An earlier refactor accidentally re-created a generator and reset
    # 'paths' to an empty list here, erasing all generated frame paths so video/metadata
    # were never produced. Removed the redundant generator + reset.
    # Optional Video
    video_path = None
    # Video frames override
    if cfg.video and cfg.video_frames > 0 and cfg.mode not in ('morph','interpolation','seed_cycle'):
        # Extend via additional seed cycle frames for video length
        extra = cfg.video_frames - len(paths)
        if extra > 0:
            base_seed = cfg.seed if cfg.seed is not None else int(time.time())
            device = pipe.unet.device
            dtype = next(pipe.unet.parameters()).dtype
            lh, lw = cfg.height//8, cfg.width//8
            in_channels = pipe.unet.config.in_channels if hasattr(pipe.unet,'config') else 4
            for i in range(extra):
                g2 = torch.Generator(device=device).manual_seed(base_seed + i*997)
                result = pipe(
                    cfg.prompt,
                    negative_prompt=cfg.negative or None,
                    num_inference_steps=cfg.steps,
                    guidance_scale=cfg.guidance,
                    height=cfg.height,
                    width=cfg.width,
                    generator=g2
                )
                img: Image.Image = result.images[0]
                fname = f"{cfg.run_id}-{len(paths)+1:03d}.png"
                fpath = os.path.join(run_dir, fname)
                utils.save_image_with_meta(img, fpath, {
                    'prompt': cfg.prompt,
                    'negative': cfg.negative,
                    'model': cfg.model,
                    'mode': cfg.mode,
                    'seed': base_seed + i*997,
                    'steps': cfg.steps,
                    'guidance': cfg.guidance,
                    'video_extend': True
                })
                paths.append(fpath)
    if cfg.video and len(paths) > 1:
        if cfg.video_target_duration and cfg.video_target_duration > 0 and cfg.video_fps <= 0:
            # grobe fps später in build_video (wenn blends berücksichtigt) -> hier 0 lassen
            fps = 0
        else:
            fps = cfg.video_fps if cfg.video_fps > 0 else (6 if len(paths) < 12 else min(30, len(paths)//2))
        vid_name = cfg.video_name or f"{cfg.run_id}.mp4"
        video_path = os.path.join(run_dir, vid_name)
        try:
            video.build_video(paths, video_path, fps, cfg.video_blend_mode, cfg.video_blend_steps, target_duration=cfg.video_target_duration)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Video fehlgeschlagen: {e}")
            video_path = None
    # Metadata
    if cfg.write_meta:
        meta = {
            'run_id': run_id,
            'mode': cfg.mode,
            'prompt': cfg.prompt,
            'negative': cfg.negative,
            'model': cfg.model,
            'steps': cfg.steps,
            'guidance': cfg.guidance,
            'height': cfg.height,
            'width': cfg.width,
            'images': len(paths),
            'seed': cfg.seed,
            'seed_cycle': cfg.seed_cycle,
            'interp_frames': cfg.interp_frames,
            'morph_frames': cfg.morph_frames,
            'files': [os.path.basename(p) for p in paths],
            'video': os.path.basename(video_path) if video_path else None
        }
        utils.write_metadata(run_dir, run_id, meta)
    return paths
