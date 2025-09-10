#!/usr/bin/env python3
"""Einfache Web UI für SDGen.

Funktionen:
- Config-Datei Auswahl (List vorhandener *.json im ./configs oder Root)
- Laden & Formular mit allen Parametern (Typbezogene Widgets)
- Output-Ordner Auswahl / Anzeige
- Start Button: führt Generation (blocking) aus
- Bild-Vorschau (512x512) mit Slider über PNGs im Run-Ordner + Auto-Play

Hinweis: UI ist bewusst minimal; für große Performanz besser Queue/Thread.
"""
import os, json, threading, time, glob
from typing import Dict, Any, List
import gradio as gr
import importlib

# --- Workaround für Gradio JSON-Schema Bug mit boolean additionalProperties ---
# Statt API komplett zu deaktivieren, säubern wir das generierte Schema rekursiv,
# indem reine bool-Werte (True/False) an Stellen ersetzt werden, wo ein Objekt
# erwartet wird. So bleibt die API verfügbar.
try:  # defensiv
    from gradio.blocks import Blocks as _Blocks  # type: ignore
    _orig_get_api_info = _Blocks.get_api_info  # type: ignore[attr-defined]
    def _sanitize(o):  # rekursive Sanitization
        if isinstance(o, bool):
            # Problemfall: True/False als Schema -> neutralisiere zu leerem Objekt
            return {}
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanitize(v) for v in o]
        return o
    def _patched_get_api_info(self):  # type: ignore[override]
        try:
            info = _orig_get_api_info(self)
            return _sanitize(info)
        except Exception:
            # Fallback: leeres Grundgerüst liefern
            return {"named_endpoints": {}, "unnamed_endpoints": []}
    _Blocks.get_api_info = _patched_get_api_info  # type: ignore
except Exception:
    pass

# Lokaler Import (Projektstruktur)
from sdgen import GenerationConfig, build_pipeline, run_generation

APP_STATE = {
    'pipe_cache': {},  # model_id -> pipeline
    'last_run_paths': [],
    'last_run_dir': None,
    'autoplay': False,
}

CONFIG_DIR_CANDIDATES = ["./", "./configs", "./config"]

PARAM_SPECS = [
    # (Feld, Typ, Min, Max, Step, Kategorie)
    ("prompt", str, None, None, None, "basic"),
    ("negative", str, None, None, None, "basic"),
    ("model", str, None, None, None, "basic"),
    ("height", int, 128, 1536, 8, "basic"),
    ("width", int, 128, 1536, 8, "basic"),
    ("steps", int, 1, 200, 1, "basic"),
    ("guidance", float, 0.0, 30.0, 0.1, "basic"),
    ("images", int, 1, 32, 1, "basic"),
    ("seed", int, 0, 2_147_483_647, 1, "basic"),
    ("half", bool, None, None, None, "perf"),
    # video
    ("video", bool, None, None, None, "video"),
    ("video_fps", int, 0, 60, 1, "video"),
    ("video_blend_mode", str, None, None, None, "video"),
    ("video_blend_steps", int, 0, 12, 1, "video"),
    ("video_target_duration", float, 0.0, 120.0, 0.5, "video"),
    ("video_frames", int, 0, 400, 1, "video"),
    # morph
    ("morph_from", str, None, None, None, "morph"),
    ("morph_to", str, None, None, None, "morph"),
    ("morph_prompts", str, None, None, None, "morph"),
    ("morph_frames", int, 0, 400, 1, "morph"),
    ("morph_latent", bool, None, None, None, "morph"),
    ("morph_seed_start", int, 0, 2_147_483_647, 1, "morph"),
    ("morph_seed_end", int, 0, 2_147_483_647, 1, "morph"),
    ("morph_slerp", bool, None, None, None, "morph"),
    ("morph_continuous", bool, None, None, None, "morph"),
    ("morph_ease", str, None, None, None, "morph"),
    ("morph_color_shift", bool, None, None, None, "morph"),
    ("morph_color_intensity", float, 0.0, 1.0, 0.01, "morph"),
    ("morph_noise_pulse", float, 0.0, 2.0, 0.01, "morph"),
    ("morph_frame_perturb", float, 0.0, 2.0, 0.01, "morph"),
    ("morph_temporal_blend", float, 0.0, 1.0, 0.01, "morph"),
    ("morph_effect_curve", str, None, None, None, "morph"),
    ("morph_smooth", bool, None, None, None, "morph"),
    # interpolation
    ("interp_seed_start", int, 0, 2_147_483_647, 1, "interp"),
    ("interp_seed_end", int, 0, 2_147_483_647, 1, "interp"),
    ("interp_frames", int, 0, 400, 1, "interp"),
    ("interp_slerp", bool, None, None, None, "interp"),
    # seed cycle
    ("seed_cycle", int, 0, 400, 1, "seed"),
    ("seed_step", int, 1, 10000, 1, "seed"),
    ("latent_jitter", float, 0.0, 1.0, 0.01, "seed"),
]

CHOICES = {
    'video_blend_mode': ['none','linear','flow'],
    'morph_ease': ['linear','ease','ease-in','ease-out','sine','quad','cubic'],
    'morph_effect_curve': ['center','linear','flat','edges'],
}

BOOLEAN_PARAMS = {name for name, t, *_ in PARAM_SPECS if t is bool}

CONFIG_FIELDS = [name for name, *_ in PARAM_SPECS]

DEFAULT_OUTPUT_DIR = 'outputs'

# ---------- Helpers ----------

def find_config_files():
    files = []
    for d in CONFIG_DIR_CANDIDATES:
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, '*.json')):
                files.append(os.path.abspath(f))
    return sorted(set(files))

def load_config(path: str) -> Dict[str, Any]:
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def list_run_dirs(base_dir: str) -> List[str]:
    """Liest alle direkten Unterordner im base_dir (z.B. outputs/<run>) und gibt absolute Pfade zurück.
    Sortiert nach Änderungszeit (neueste zuerst)."""
    if not base_dir or not os.path.isdir(base_dir):
        return []
    entries = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            entries.append((os.path.getmtime(full), full))
    entries.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in entries]

# ---------- Generation ----------

def get_pipeline(model_id: str, half: bool):
    key = (model_id, half)
    pipe = APP_STATE['pipe_cache'].get(key)
    if pipe is None:
        pipe = build_pipeline(model_id, half)
        APP_STATE['pipe_cache'][key] = pipe
    return pipe

def run_generation_thread(cfg_dict: Dict[str, Any]):
    try:
        cfg = GenerationConfig(**cfg_dict)
        pipe = get_pipeline(cfg.model, cfg.half)
        paths = run_generation(cfg, pipe)
        APP_STATE['last_run_paths'] = paths
        APP_STATE['last_run_dir'] = cfg.run_dir
    except Exception as e:
        APP_STATE['last_run_paths'] = []
        APP_STATE['last_run_dir'] = None
        APP_STATE['last_error'] = str(e)

# ---------- UI dynamic form ----------

def build_param_components():
    comps = {}
    with gr.Accordion("Basis", open=True):
        comps['prompt'] = gr.Textbox(label='Prompt')
        comps['negative'] = gr.Textbox(label='Negative Prompt')
        comps['model'] = gr.Textbox(value='stabilityai/sd-turbo', label='Model ID')
        comps['height'] = gr.Slider(128,1536,step=8,value=512,label='Height')
        comps['width'] = gr.Slider(128,1536,step=8,value=512,label='Width')
        comps['steps'] = gr.Slider(1,200,step=1,value=6,label='Steps')
        comps['guidance'] = gr.Slider(0,30,step=0.1,value=0.0,label='Guidance')
        comps['images'] = gr.Slider(1,32,step=1,value=1,label='Images')
        comps['seed'] = gr.Number(value=None,label='Seed (leer = random)')
        comps['half'] = gr.Checkbox(value=False,label='FP16 (half)')
    with gr.Accordion("Video", open=False):
        comps['video'] = gr.Checkbox(label='Video aktiv')
        comps['video_fps'] = gr.Slider(0,60,step=1,value=0,label='FPS (0=auto)')
        comps['video_blend_mode'] = gr.Dropdown(CHOICES['video_blend_mode'], value='none', label='Blend Mode')
        comps['video_blend_steps'] = gr.Slider(0,12,step=1,value=0,label='Blend Steps')
        comps['video_target_duration'] = gr.Slider(0,120,step=0.5,value=0,label='Target Duration (s)')
        comps['video_frames'] = gr.Slider(0,400,step=1,value=0,label='Force Frames (single/batch)')
    with gr.Accordion("Morph", open=False):
        comps['morph_from'] = gr.Textbox(label='Morph From')
        comps['morph_to'] = gr.Textbox(label='Morph To')
        comps['morph_prompts'] = gr.Textbox(label='Morph Prompts (comma list)')
        comps['morph_frames'] = gr.Slider(0,400,step=1,value=0,label='Morph Frames')
        comps['morph_latent'] = gr.Checkbox(label='Latent Morph')
        comps['morph_seed_start'] = gr.Number(value=None,label='Morph Seed Start')
        comps['morph_seed_end'] = gr.Number(value=None,label='Morph Seed End')
        comps['morph_slerp'] = gr.Checkbox(label='Morph Slerp')
        comps['morph_continuous'] = gr.Checkbox(label='Continuous Flow')
        comps['morph_ease'] = gr.Dropdown(CHOICES['morph_ease'], value='linear', label='Ease')
        comps['morph_color_shift'] = gr.Checkbox(label='Color Shift')
        comps['morph_color_intensity'] = gr.Slider(0,1,step=0.01,value=0.25,label='Color Intensity')
        comps['morph_noise_pulse'] = gr.Slider(0,2,step=0.01,value=0,label='Noise Pulse')
        comps['morph_frame_perturb'] = gr.Slider(0,2,step=0.01,value=0,label='Frame Perturb')
        comps['morph_temporal_blend'] = gr.Slider(0,1,step=0.01,value=0,label='Temporal Blend')
        comps['morph_effect_curve'] = gr.Dropdown(CHOICES['morph_effect_curve'], value='center', label='Effect Curve')
        comps['morph_smooth'] = gr.Checkbox(label='Smooth')
    with gr.Accordion("Interpolation", open=False):
        comps['interp_seed_start'] = gr.Number(value=None,label='Seed Start')
        comps['interp_seed_end'] = gr.Number(value=None,label='Seed End')
        comps['interp_frames'] = gr.Slider(0,400,step=1,value=0,label='Frames')
        comps['interp_slerp'] = gr.Checkbox(label='Slerp')
    with gr.Accordion("Seed Cycle", open=False):
        comps['seed_cycle'] = gr.Slider(0,400,step=1,value=0,label='Cycle Count')
        comps['seed_step'] = gr.Slider(1,10000,step=1,value=997,label='Seed Step')
        comps['latent_jitter'] = gr.Slider(0,1,step=0.01,value=0,label='Latent Jitter')
    return comps

# ---------- Preview Handling ----------

def list_pngs(run_dir: str) -> List[str]:
    if not run_dir or not os.path.isdir(run_dir):
        return []
    return sorted([p for p in glob.glob(os.path.join(run_dir, '*.png'))])

_preview_index = 0

def get_preview_image(idx: int) -> Any:
    if not APP_STATE['last_run_dir']:
        return None
    files = list_pngs(APP_STATE['last_run_dir'])
    if not files:
        return None
    i = max(0, min(idx, len(files)-1))
    return files[i]

# ---------- Gradio Callbacks ----------

def refresh_configs():
    return gr.update(choices=find_config_files())

def load_selected_config(path):
    if not path:
        return {k: None for k in CONFIG_FIELDS}
    try:
        data = load_config(path)
    except Exception:
        return {k: None for k in CONFIG_FIELDS}
    # Flatten list morph_prompts to comma str
    if isinstance(data.get('morph_prompts'), list):
        data['morph_prompts'] = ', '.join(data['morph_prompts'])
    return data

def run_button_click(output_dir, values: Dict[str, Any]):
    # Build cfg dict from component values
    cfg = {}
    for name in CONFIG_FIELDS:
        if name not in values:
            continue
        v = values[name]
        if name == 'morph_prompts' and isinstance(v, str) and v.strip():
            cfg['morph_prompts'] = [s.strip() for s in v.split(',') if s.strip()]
        elif name in BOOLEAN_PARAMS:
            if v:
                cfg[name] = True
        else:
            if v not in (None, ''):
                cfg[name] = v
    if 'prompt' not in cfg and not cfg.get('morph_prompts'):
        return gr.update(value="Fehlender Prompt oder morph_prompts"), None
    cfg.setdefault('outdir', output_dir or 'outputs')
    # Launch in thread
    th = threading.Thread(target=run_generation_thread, args=(cfg,), daemon=True)
    th.start()
    return gr.update(value="Running..."), None

def update_preview(index_slider):
    img_path = get_preview_image(int(index_slider))
    return img_path

def autoplay_tick(current_index, autoplay, delay):
    if not autoplay:
        return current_index
    files = list_pngs(APP_STATE['last_run_dir'])
    if not files:
        return 0
    nxt = (current_index + 1) % len(files)
    return nxt

def refresh_run_dirs(outdir: str):
    dirs = list_run_dirs(outdir)
    return gr.update(choices=dirs, value=(dirs[0] if dirs else None))

def select_run_dir(run_dir: str):
    APP_STATE['last_run_dir'] = run_dir or None
    files = list_pngs(run_dir) if run_dir else []
    max_index = max(0, len(files)-1)
    first = files[0] if files else None
    # Rückgabe: Slider Update, Preview Image
    return gr.update(max=max_index, value=0), first

# ---------- Build UI ----------

def build_ui():
    with gr.Blocks(title="SDGen Web UI") as demo:
        gr.Markdown("# SDGen Web UI")
        with gr.Row():
            with gr.Column(scale=1):
                config_box = gr.Dropdown(label='Config Datei', choices=find_config_files(), interactive=True)
                refresh_btn = gr.Button('Configs aktualisieren')
                load_btn = gr.Button('Laden')
                output_dir = gr.Textbox(label='Output Dir', value=DEFAULT_OUTPUT_DIR)
                run_dir_select = gr.Dropdown(label='Run Ordner (Unterordner)', choices=list_run_dirs(DEFAULT_OUTPUT_DIR), interactive=True)
                run_dirs_refresh = gr.Button('Run Ordner aktualisieren')
                status = gr.Textbox(label='Status / Log', interactive=False)
            with gr.Column(scale=2):
                comps = build_param_components()
            with gr.Column(scale=1):
                preview = gr.Image(label='Preview', type='filepath')
                index_slider = gr.Slider(0,0,step=1,value=0,label='Frame Index')
                autoplay = gr.Checkbox(label='Auto Play')
                delay = gr.Slider(0.1,2.0,step=0.1,value=0.5,label='Auto Play Delay (s)')
                run_btn = gr.Button('Generate', variant='primary')
        # Laden Helper
        comp_order = list(comps.keys())
        def _load_wrapper(path):
            data = load_selected_config(path)
            return [data.get(name) for name in comp_order]
        # Events
        refresh_btn.click(fn=refresh_configs, outputs=config_box)
        load_btn.click(fn=_load_wrapper, inputs=config_box, outputs=[comps[n] for n in comps])
        run_btn.click(
            fn=lambda outdir, *vals: run_button_click(outdir, {k: v for k, v in zip(comps.keys(), vals)}),
            inputs=[output_dir] + [comps[n] for n in comps],
            outputs=[status, preview]
        )
        index_slider.change(fn=update_preview, inputs=index_slider, outputs=preview)
        autoplay.change(fn=lambda x: x, inputs=autoplay, outputs=autoplay)
        run_dirs_refresh.click(fn=refresh_run_dirs, inputs=output_dir, outputs=run_dir_select)
        output_dir.change(fn=refresh_run_dirs, inputs=output_dir, outputs=run_dir_select)
        run_dir_select.change(fn=select_run_dir, inputs=run_dir_select, outputs=[index_slider, preview])
        demo.load(fn=lambda: 0, outputs=index_slider)
        gr.Markdown("Hinweis: Auto Play aktualisiert aktuell nicht automatisch ohne manuelles Refresh (vereinfachte Implementierung).")
        demo.queue()
        return demo

if __name__ == '__main__':
    ui = build_ui()
    # Deaktiviert API Schema Anzeige, um bekannten Bug (TypeError in json_schema_to_python_type) bei 4.31.0 zu umgehen
    ui.launch(show_api=False)
