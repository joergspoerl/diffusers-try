#!/usr/bin/env python3
"""Haupt-CLI für SDGen."""
import argparse, json, os, sys
from dataclasses import asdict
from sdgen import GenerationConfig, build_pipeline, run_generation

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', help='Prompt (optional wenn --config genutzt)')
    p.add_argument('--config', help='JSON Konfigurationsdatei als Eingabe')
    p.add_argument('--resume', help='Fortsetzen/Resume aus bestehendem Output-Verzeichnis')
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
    p.add_argument('--morph-continuous', action='store_true', help='Kontinuierlicher Multi-Prompt Morph (kein Segment-Reset)')
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
    
    # Resume Modus: Lade Config aus bestehendem Verzeichnis
    if args.resume:
        resume_dir = args.resume
        if not os.path.exists(resume_dir):
            raise SystemExit(f'Resume-Verzeichnis existiert nicht: {resume_dir}')
        
        # Suche nach Config-Datei im Verzeichnis
        config_files = [f for f in os.listdir(resume_dir) if f.endswith('-config.json')]
        if not config_files:
            raise SystemExit(f'Keine Config-Datei (*-config.json) in {resume_dir} gefunden')
        
        config_path = os.path.join(resume_dir, config_files[0])
        print(f'[INFO] Lade Config aus: {config_path}')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_data = json.load(f)
        
        # Überschreibe run_dir und run_id aus dem Resume-Verzeichnis
        cfg_data['run_dir'] = resume_dir
        cfg_data['run_id'] = os.path.basename(resume_dir)
        cfg_data['make_run_dir'] = False  # Verzeichnis existiert bereits
        
        # CLI-Overrides für Video-Parameter erlauben
        argv_tokens = set(sys.argv[1:])
        def flag_present(opt: str) -> bool:
            return opt in argv_tokens
        def set_if(name, value, opt=None, allow_zero=True):
            if opt and not flag_present(opt):
                return
            if value is None:
                return
            if not allow_zero and value == 0:
                return
            cfg_data[name] = value
        def set_bool(name, value, opt):
            if value and flag_present(opt):
                cfg_data[name] = True
        
        # Erlaube Video-Parameter zu überschreiben
        set_bool('video', args.video, '--video')
        set_if('video_name', args.video_name, '--video-name')
        set_if('video_fps', args.video_fps, '--video-fps')
        set_if('video_blend_mode', args.video_blend_mode, '--video-blend-mode')
        set_if('video_blend_steps', args.video_blend_steps, '--video-blend-steps')
        set_if('video_target_duration', args.video_target_duration, '--video-target-duration')
        
        print(f'[INFO] Resume-Modus: Verwende existierende Frames aus {resume_dir}')
        
    else:
        # Normaler Modus: Basis Konfiguration (evtl. aus Datei)
        if args.config:
            with open(args.config,'r',encoding='utf-8') as f:
                cfg_data = json.load(f)
        else:
            cfg_data = {}
    # CLI Overrides: nur wenn explizit gesetzt
    argv_tokens = set(sys.argv[1:])
    def flag_present(opt: str) -> bool:
        return opt in argv_tokens
    def set_if(name, value, opt=None, allow_zero=True):
        # opt: zugehöriger CLI-Option-Name (z.B. '--morph-frames')
        if opt and not flag_present(opt):
            return  # nicht explizit gesetzt
        if value is None:
            return
        if not allow_zero and value == 0:
            return
        cfg_data[name] = value
    def set_bool(name, value, opt):
        # store_true Flags: nur setzen wenn True (explizit angegeben)
        if value and flag_present(opt):
            cfg_data[name] = True
        # Falls in config True und Flag NICHT gesetzt -> belassen
    # Primitive Felder
    set_if('prompt', args.prompt, '--prompt')
    set_if('negative', args.negative, '--negative')
    set_if('model', args.model, '--model')
    set_if('height', args.height, '--height')
    set_if('width', args.width, '--width')
    set_if('steps', args.steps, '--steps')
    set_if('guidance', args.guidance, '--guidance')
    set_if('images', args.images, '--images')
    set_if('seed', args.seed, '--seed')
    set_bool('half', args.half, '--half')
    set_if('outdir', args.outdir, '--outdir')
    # Morph
    set_if('morph_from', args.morph_from, '--morph-from')
    set_if('morph_to', args.morph_to, '--morph-to')
    if args.morph_prompts and flag_present('--morph-prompts'):
        cfg_data['morph_prompts'] = [s.strip() for s in args.morph_prompts.split(',')]
    set_if('morph_frames', args.morph_frames, '--morph-frames')
    set_bool('morph_latent', args.morph_latent, '--morph-latent')
    set_if('morph_seed_start', args.morph_seed_start, '--morph-seed-start')
    set_if('morph_seed_end', args.morph_seed_end, '--morph-seed-end')
    set_bool('morph_slerp', args.morph_slerp, '--morph-slerp')
    set_bool('morph_continuous', args.morph_continuous, '--morph-continuous')
    set_if('morph_ease', args.morph_ease, '--morph-ease')
    set_bool('morph_color_shift', args.morph_color_shift, '--morph-color-shift')
    set_if('morph_color_intensity', args.morph_color_intensity, '--morph-color-intensity')
    set_if('morph_noise_pulse', args.morph_noise_pulse, '--morph-noise-pulse')
    set_if('morph_frame_perturb', args.morph_frame_perturb, '--morph-frame-perturb')
    set_if('morph_temporal_blend', args.morph_temporal_blend, '--morph-temporal-blend')
    set_if('morph_effect_curve', args.morph_effect_curve, '--morph-effect-curve')
    set_bool('morph_smooth', args.morph_smooth, '--morph-smooth')
    # Seed cycle / interpolation / video
    set_if('seed_cycle', args.seed_cycle, '--seed-cycle')
    set_if('seed_step', args.seed_step, '--seed-step')
    set_if('latent_jitter', args.latent_jitter, '--latent-jitter')
    set_bool('video', args.video, '--video')
    set_if('video_name', args.video_name, '--video-name')
    set_if('video_fps', args.video_fps, '--video-fps')
    set_if('video_blend_mode', args.video_blend_mode, '--video-blend-mode')
    set_if('video_blend_steps', args.video_blend_steps, '--video-blend-steps')
    set_if('video_target_duration', args.video_target_duration, '--video-target-duration')
    set_if('video_frames', args.video_frames, '--video-frames')
    set_if('interp_seed_start', args.interp_seed_start, '--interp-seed-start')
    set_if('interp_seed_end', args.interp_seed_end, '--interp-seed-end')
    set_if('interp_frames', args.interp_frames, '--interp-frames')
    set_bool('interp_slerp', args.interp_slerp, '--interp-slerp')
    set_bool('cpu_offload', args.cpu_offload, '--cpu-offload')
    set_bool('seq_offload', args.seq_offload, '--seq-offload')
    set_bool('no_slicing', args.no_slicing, '--no-slicing')
    set_bool('info_only', args.info, '--info')
    # Validierung: Prompt muss vorhanden sein (außer info_only) wenn kein Morph-Liste generiert
    if not cfg_data.get('prompt') and not cfg_data.get('morph_prompts') and not (cfg_data.get('morph_from') and cfg_data.get('morph_to')) and not cfg_data.get('info_only'):
        raise SystemExit('Fehlender Prompt: --prompt oder Morph-Parameter oder --config mit prompt angeben')
    cfg = GenerationConfig(**cfg_data)
    if args.config:
        print('[DEBUG] Effektive Konfiguration nach Merge:')
        try:
            print(json.dumps(cfg_data, indent=2, ensure_ascii=False))
        except Exception:
            pass
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
