from typing import List
import os, time, torch
from dataclasses import asdict
from PIL import Image
from .config import GenerationConfig
from .pipeline import infer_defaults
from . import utils
from . import interpolate, seedcycle, morph, video
import signal
import sys

# Placeholder: This module would be filled by migrating logic stepwise from generate.py

# Global variables for graceful shutdown
interrupted = False
current_paths = []
current_cfg = None
current_run_dir = None

def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\n[INFO] Strg+C erkannt! Beende Bildgenerierung und erstelle Video aus vorhandenen Frames...")

def run_generation(cfg: GenerationConfig, pipe) -> List[str]:
    global current_paths, current_cfg, current_run_dir, interrupted
    
    # Signal handler für Strg+C registrieren
    signal.signal(signal.SIGINT, signal_handler)
    
    steps, guidance = infer_defaults(cfg.model, cfg.steps, cfg.guidance)
    overall_start = time.time()
    
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
    
    # Global variables für signal handler setzen
    current_cfg = cfg
    current_run_dir = run_dir
    current_paths = []
    
    # Konfiguration SOFORT wegschreiben (außer im Resume-Modus)
    if cfg.make_run_dir:  # Nur wenn es ein neues Verzeichnis ist
        try:
            config_path = os.path.join(run_dir, f"{run_id}-config.json")
            with open(config_path,'w',encoding='utf-8') as f:
                import json
                json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
            print(f"[INFO] Konfiguration gespeichert: {config_path}")
        except Exception as e:
            print(f'[WARN] Konnte Config nicht schreiben: {e}')
    
    # Prüfe auf existierende Frames im Resume-Modus
    existing_frames = []
    if not cfg.make_run_dir:  # Resume-Modus
        print(f"[INFO] Suche nach existierenden Frames in {run_dir}...")
        import glob
        pattern = os.path.join(run_dir, f"{run_id}-*.png")
        existing_frames = sorted(glob.glob(pattern))
        if existing_frames:
            print(f"[INFO] Gefunden: {len(existing_frames)} existierende Frames")
            for frame in existing_frames[:5]:  # Zeige erste 5
                print(f"  - {os.path.basename(frame)}")
            if len(existing_frames) > 5:
                print(f"  ... und {len(existing_frames)-5} weitere")
        else:
            print("[INFO] Keine existierenden Frames gefunden")
            
        # Prüfe auf existierendes Video
        video_pattern = os.path.join(run_dir, "*.mp4")
        existing_videos = glob.glob(video_pattern)
        if existing_videos:
            print(f"[INFO] Existierende Videos gefunden: {[os.path.basename(v) for v in existing_videos]}")
    
    print(f"[INFO] Konfiguration geladen")
    # Decide mode
    paths: List[str] = []
    # Fill defaults
    cfg.steps, cfg.guidance = infer_defaults(cfg.model, cfg.steps, cfg.guidance)

    # Interpolation priority after morph
    if (cfg.morph_prompts and len(cfg.morph_prompts) >= 2 and cfg.morph_frames > 1) or (cfg.morph_from and cfg.morph_to and cfg.morph_frames > 1):
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
        vid_name = cfg.video_name or f"{cfg.run_id}.mp4"
        video_path = os.path.join(run_dir, vid_name)
        
        # Prüfe ob Video bereits existiert (außer bei expliziter Video-Parameter-Änderung)
        if os.path.exists(video_path) and not cfg.make_run_dir:
            print(f"[INFO] Video bereits vorhanden: {video_path}")
            # Prüfe ob Video-Parameter geändert wurden
            argv_tokens = set(sys.argv[1:])
            video_flags = ['--video-fps', '--video-blend-mode', '--video-blend-steps', '--video-target-duration']
            video_params_changed = any(flag in argv_tokens for flag in video_flags)
            
            if not video_params_changed:
                print("[INFO] Keine Video-Parameter geändert, überspringe Video-Erstellung")
            else:
                print("[INFO] Video-Parameter geändert, erstelle neues Video...")
                if cfg.video_target_duration and cfg.video_target_duration > 0 and cfg.video_fps <= 0:
                    fps = 0
                else:
                    fps = cfg.video_fps if cfg.video_fps > 0 else (6 if len(paths) < 12 else min(30, len(paths)//2))
                try:
                    video.build_video(paths, video_path, fps, cfg.video_blend_mode, cfg.video_blend_steps, target_duration=cfg.video_target_duration)
                    print(f"[INFO] Neues Video erstellt: {video_path}")
                except Exception as e:
                    print(f"[WARN] Video fehlgeschlagen: {e}")
                    video_path = None
        else:
            # Neues Video erstellen
            if cfg.video_target_duration and cfg.video_target_duration > 0 and cfg.video_fps <= 0:
                fps = 0
            else:
                fps = cfg.video_fps if cfg.video_fps > 0 else (6 if len(paths) < 12 else min(30, len(paths)//2))
            try:
                if interrupted:
                    print(f"[INFO] Erstelle Video aus {len(paths)} vorhandenen Frames...")
                video.build_video(paths, video_path, fps, cfg.video_blend_mode, cfg.video_blend_steps, target_duration=cfg.video_target_duration)
                if interrupted:
                    print(f"[INFO] Video erfolgreich erstellt: {video_path}")
            except Exception as e:  # pragma: no cover
                print(f"[WARN] Video fehlgeschlagen: {e}")
                video_path = None
    elif interrupted and len(paths) > 0:
        print(f"[INFO] {len(paths)} Frames gespeichert, aber Video-Erstellung übersprungen (cfg.video=False)")
        
    # Metadata auch bei Interrupt schreiben
    duration = time.time() - overall_start
    if cfg.write_meta or interrupted:
        meta = {
            'run_id': run_id,
            'mode': cfg.mode,
            'prompt': cfg.prompt,
            'morph_prompts': cfg.morph_prompts,
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
            'video': os.path.basename(video_path) if video_path else None,
            'runtime_seconds': round(duration, 3),
            'avg_seconds_per_image': round(duration/len(paths), 3) if paths else None,
            'interrupted': interrupted,
            'completed_frames': len(paths),
            'total_planned_frames': cfg.morph_frames if hasattr(cfg, 'morph_frames') else len(paths)
        }
        utils.write_metadata(run_dir, run_id, meta)
        # Markdown Summary
        try:
            summary_path = os.path.join(run_dir, f"{run_id}-summary.md")
            cfg_dict = asdict(cfg)
            def fmt(val):
                if isinstance(val, list):
                    return ', '.join(str(v) for v in val)
                return val
            core_rows = [
                ('Run ID', run_id),
                ('Mode', cfg.mode),
                ('Model', cfg.model),
                ('Prompt', (cfg.prompt if cfg.prompt else '')), 
                ('Morph Prompts', fmt(cfg.morph_prompts) if cfg.morph_prompts else ''),
                ('Images', len(paths)),
                ('Height', cfg.height),
                ('Width', cfg.width),
                ('Steps', cfg.steps),
                ('Guidance', cfg.guidance),
                ('Seed', cfg.seed),
                ('Seed Cycle', cfg.seed_cycle),
                ('Interpolation Frames', cfg.interp_frames),
                ('Morph Frames', cfg.morph_frames),
                ('Video', os.path.basename(video_path) if video_path else ''),
                ('Runtime (s)', round(duration,3)),
                ('Avg / Image (s)', round(duration/len(paths),3) if paths else 'n/a'),
                ('Interrupted', 'Yes' if interrupted else 'No'),
                ('Completed Frames', len(paths)),
                ('Total Planned', cfg.morph_frames if hasattr(cfg, 'morph_frames') else len(paths))
            ]
            lines = [f"# Run Summary: {run_id}", '', '## Parameter', '', '| Key | Value |', '|-----|-------|']
            for k,v in core_rows:
                lines.append(f"| {k} | {v} |")
            lines += ['', '## Files', '']
            for pth in paths:
                lines.append(f"- {os.path.basename(pth)}")
            if video_path:
                lines.append(f"- {os.path.basename(video_path)} (video)")
            lines += ['', '## Raw Config (excerpt)', '', '```json']
            # Limit raw config size and remove large internals
            raw_cfg = {k: v for k, v in cfg_dict.items() if k not in ('run_dir',)}
            import json as _json
            lines.append(_json.dumps(raw_cfg, indent=2, ensure_ascii=False))
            lines.append('```')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        except Exception as e:  # pragma: no cover
            print('[WARN] Konnte Summary Markdown nicht schreiben:', e)
    return paths
