#!/usr/bin/env python3
"""Modularer Einstiegspunkt: nutzt sdgen-Paket."""
import argparse, json
from sdgen import GenerationConfig, build_pipeline, run_generation

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', required=True)
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
    cfg = GenerationConfig(
        prompt=args.prompt,
        negative=args.negative,
        model=args.model,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        images=args.images,
        seed=args.seed,
        half=args.half,
        outdir=args.outdir
    ,morph_from=args.morph_from
    ,morph_to=args.morph_to
    ,morph_frames=args.morph_frames
    ,morph_latent=args.morph_latent
    ,morph_seed_start=args.morph_seed_start
    ,morph_seed_end=args.morph_seed_end
    ,morph_slerp=args.morph_slerp
    ,morph_ease=args.morph_ease
    ,morph_color_shift=args.morph_color_shift
    ,morph_color_intensity=args.morph_color_intensity
    ,morph_noise_pulse=args.morph_noise_pulse
    ,morph_frame_perturb=args.morph_frame_perturb
    ,morph_temporal_blend=args.morph_temporal_blend
    ,morph_effect_curve=args.morph_effect_curve
    ,morph_smooth=args.morph_smooth
    ,seed_cycle=args.seed_cycle
    ,seed_step=args.seed_step
    ,latent_jitter=args.latent_jitter
    ,video=args.video
    ,video_name=args.video_name
    ,video_fps=args.video_fps
    ,video_blend_mode=args.video_blend_mode
    ,video_blend_steps=args.video_blend_steps
    ,video_target_duration=args.video_target_duration
    ,video_frames=args.video_frames
    ,interp_seed_start=args.interp_seed_start
    ,interp_seed_end=args.interp_seed_end
    ,interp_frames=args.interp_frames
    ,interp_slerp=args.interp_slerp
    ,cpu_offload=args.cpu_offload
    ,seq_offload=args.seq_offload
    ,no_slicing=args.no_slicing
    ,info_only=args.info
    )
    pipe = build_pipeline(cfg.model, cfg.half)
    paths = run_generation(cfg, pipe)
    print('Dateien:')
    for p in paths:
        print(' -', p)
    print(json.dumps({'count': len(paths)}, indent=2))

if __name__ == '__main__':
    main()
