#!/usr/bin/env python3
"""Haupt-CLI für SDGen."""
import argparse, json, os
from dataclasses import asdict
from sdgen import GenerationConfig, build_pipeline, run_generation

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', help='Prompt (optional wenn --config genutzt)')
    p.add_argument('--config', help='JSON Konfigurationsdatei als Eingabe')
    p.add_argument('--negative', default='')
    p.add_argument('--model', default='stabilityai/sd-turbo')
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--steps', type=int)
    p.add_argument('--guidance', type=float)
    p.add_argument('--images', type=int, default=1)
    p.add_argument('--seed', type=int)
    p.add_argument('--outdir', default='outputs')
    p.add_argument('--half', action='store_true')
    p.add_argument('--cpu-offload', action='store_true')
    p.add_argument('--seq-offload', action='store_true')
    p.add_argument('--no-slicing', action='store_true')
    p.add_argument('--info', action='store_true')
    # Seed cycle & jitter
    p.add_argument('--seed-cycle', type=int, default=0)
    p.add_argument('--seed-step', type=int, default=997)
    p.add_argument('--latent-jitter', type=float, default=0.0)
    # Video options
    p.add_argument('--video', action='store_true')
    p.add_argument('--video-name')
    p.add_argument('--video-fps', type=int, default=0)
    p.add_argument('--video-blend-mode', choices=['none','linear','flow'], default='none')
    p.add_argument('--video-blend-steps', type=int, default=0)
    p.add_argument('--video-target-duration', type=float, default=0.0)
    p.add_argument('--video-frames', type=int, default=0)
    # Morph / advanced
    p.add_argument('--morph-from')
    p.add_argument('--morph-to')
    p.add_argument('--morph-prompts', help='Kommagetrennte Liste für Sequenz-Morph (überschreibt --morph-from/--morph-to)')
    p.add_argument('--morph-frames', type=int, default=0)
    p.add_argument('--morph-latent', action='store_true')
    p.add_argument('--morph-seed-start', type=int)
    p.add_argument('--morph-seed-end', type=int)
    p.add_argument('--morph-slerp', action='store_true')
    # Interpolation
    p.add_argument('--interp-seed-start', type=int)
    p.add_argument('--interp-seed-end', type=int)
    p.add_argument('--interp-frames', type=int, default=0)
    p.add_argument('--interp-slerp', action='store_true')
    p.add_argument('--morph-ease', default='linear')
    p.add_argument('--morph-color-shift', action='store_true')
    p.add_argument('--morph-color-intensity', type=float, default=0.25)
    p.add_argument('--morph-noise-pulse', type=float, default=0.0)
    p.add_argument('--morph-frame-perturb', type=float, default=0.0)
    p.add_argument('--morph-temporal-blend', type=float, default=0.0)
    p.add_argument('--morph-effect-curve', default='center')
    p.add_argument('--morph-smooth', action='store_true')
    return p.parse_args()

def main():
    args = parse()
    # Basis Konfiguration (evtl. aus Datei)
    if args.config:
        with open(args.config,'r',encoding='utf-8') as f:
            cfg_data = json.load(f)
    else:
        cfg_data = {}
    # CLI Overrides (überschreiben Einträge aus Datei)
    def set_if(name, value):
        if value is not None:
            cfg_data[name] = value
    # Primitive Felder
    set_if('prompt', args.prompt)
    set_if('negative', args.negative)
    set_if('model', args.model)
    set_if('height', args.height)
    set_if('width', args.width)
    set_if('steps', args.steps)
    set_if('guidance', args.guidance)
    set_if('images', args.images)
    set_if('seed', args.seed)
    set_if('half', args.half)
    set_if('outdir', args.outdir)
    # Morph
    set_if('morph_from', args.morph_from)
    set_if('morph_to', args.morph_to)
    if args.morph_prompts:
        cfg_data['morph_prompts'] = [s.strip() for s in args.morph_prompts.split(',')]
    set_if('morph_frames', args.morph_frames)
    set_if('morph_latent', args.morph_latent)
    set_if('morph_seed_start', args.morph_seed_start)
    set_if('morph_seed_end', args.morph_seed_end)
    set_if('morph_slerp', args.morph_slerp)
    set_if('morph_ease', args.morph_ease)
    set_if('morph_color_shift', args.morph_color_shift)
    set_if('morph_color_intensity', args.morph_color_intensity)
    set_if('morph_noise_pulse', args.morph_noise_pulse)
    set_if('morph_frame_perturb', args.morph_frame_perturb)
    set_if('morph_temporal_blend', args.morph_temporal_blend)
    set_if('morph_effect_curve', args.morph_effect_curve)
    set_if('morph_smooth', args.morph_smooth)
    # Seed cycle / interpolation / video
    set_if('seed_cycle', args.seed_cycle)
    set_if('seed_step', args.seed_step)
    set_if('latent_jitter', args.latent_jitter)
    set_if('video', args.video)
    set_if('video_name', args.video_name)
    set_if('video_fps', args.video_fps)
    set_if('video_blend_mode', args.video_blend_mode)
    set_if('video_blend_steps', args.video_blend_steps)
    set_if('video_target_duration', args.video_target_duration)
    set_if('video_frames', args.video_frames)
    set_if('interp_seed_start', args.interp_seed_start)
    set_if('interp_seed_end', args.interp_seed_end)
    set_if('interp_frames', args.interp_frames)
    set_if('interp_slerp', args.interp_slerp)
    set_if('cpu_offload', args.cpu_offload)
    set_if('seq_offload', args.seq_offload)
    set_if('no_slicing', args.no_slicing)
    set_if('info_only', args.info)
    # Validierung: Prompt muss vorhanden sein (außer info_only) wenn kein Morph-Liste generiert
    if not cfg_data.get('prompt') and not cfg_data.get('morph_prompts') and not (cfg_data.get('morph_from') and cfg_data.get('morph_to')) and not cfg_data.get('info_only'):
        raise SystemExit('Fehlender Prompt: --prompt oder Morph-Parameter oder --config mit prompt angeben')
    cfg = GenerationConfig(**cfg_data)
    pipe = build_pipeline(cfg.model, cfg.half)
    paths = run_generation(cfg, pipe)
    print('Dateien:')
    for p in paths:
        print(' -', p)
    # Speichere vollständige Config als separate Datei (optional zusätzlich zur Run-Metadatei)
    if cfg.run_dir and cfg.run_id:
        full_cfg_path = os.path.join(cfg.run_dir, f"{cfg.run_id}-config.json")
        try:
            with open(full_cfg_path,'w',encoding='utf-8') as f:
                json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
            print(f"Konfiguration gespeichert: {full_cfg_path}")
        except Exception as e:
            print('[WARN] Konnte vollständige Config nicht schreiben:', e)
    print(json.dumps({'count': len(paths)}, indent=2))

if __name__ == '__main__':  # pragma: no cover
    main()
