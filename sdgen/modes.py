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
                img.save(fpath)
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
            img.save(fpath)
            paths.append(fpath)
    # NOTE: (Bugfix) An earlier refactor accidentally re-created a generator and reset
    # 'paths' to an empty list here, erasing all generated frame paths so video/metadata
    # were never produced. Removed the redundant generator + reset.
    # Optional Video
    video_path = None
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
