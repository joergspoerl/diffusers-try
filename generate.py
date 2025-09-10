#!/usr/bin/env python3
"""Ein einfaches Beispiel-Script zur Bildgenerierung mit Hugging Face diffusers.

Funktionen:
- Text-zu-Bild mit Stable Diffusion Turbo (schnell, 1 Schritt) oder klassisch (mehr Qualität)
- Optionale Negative Prompts
- Steuerung von Größe, Seed und Anzahl der Bilder

Voraussetzungen:
    pip install -r requirements.txt
    (Optional) Hugging Face Login falls ein Modell Auth braucht: `huggingface-cli login`

Beispiele:
    python generate.py --prompt "ein fotorealistischer roter Oldtimer vor Bergpanorama" \
        --model stabilityai/sd-turbo --steps 2 --guidance 0.0

    python generate.py --prompt "studio lighting portrait of a golden retriever" \
        --model runwayml/stable-diffusion-v1-5 --steps 25 --guidance 7.5

Ausgabe:
    Legt PNG-Dateien unter ./outputs an.
"""
from __future__ import annotations
import argparse
import os
import time
import json
import platform
from dataclasses import dataclass
from typing import Optional, List
import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from PIL import Image


DEFAULT_MODEL_TURBO = "stabilityai/sd-turbo"
DEFAULT_MODEL_SD15 = "runwayml/stable-diffusion-v1-5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-zu-Bild mit diffusers")
    p.add_argument("--prompt", required=True, help="Eingabe Prompt")
    p.add_argument("--negative", default="", help="Optionaler Negativ-Prompt")
    p.add_argument("--model", default=DEFAULT_MODEL_TURBO, help="HF Model-ID")
    p.add_argument("--height", type=int, default=512, help="Bildhöhe")
    p.add_argument("--width", type=int, default=512, help="Bildbreite")
    p.add_argument("--steps", type=int, default=None, help="Anzahl der Inference Steps (auto wenn leer)")
    p.add_argument("--guidance", type=float, default=None, help="Classifier Free Guidance Scale (Turbo: 0.0–1.5, SD1.5: 5–9)")
    p.add_argument("--images", type=int, default=1, help="Wie viele Bilder erzeugen")
    p.add_argument("--seed", type=int, default=None, help="Seed für Reproduzierbarkeit")
    p.add_argument("--outdir", default="outputs", help="Ausgabeverzeichnis")
    p.add_argument("--half", action="store_true", help="FP16 erzwingen wenn möglich")
    p.add_argument("--cpu-offload", action="store_true", help="Model CPU Offload (falls GPU vorhanden, reduziert VRAM)")
    p.add_argument("--seq-offload", action="store_true", help="Sequentielles Offload (noch weniger VRAM, langsamer)")
    p.add_argument("--no-slicing", action="store_true", help="Attention Slicing deaktivieren")
    p.add_argument("--info", action="store_true", help="Nur Model-Info (Parameter / Komponenten) ausgeben und beenden")
    p.add_argument("--video", metavar="DATEI", help="Erzeuge ein kurzes MP4 aus einer Sequenz (nutzt leichte Seed-Variation)")
    p.add_argument("--video-frames", type=int, default=0, help="Anzahl Frames fürs Video (überschreibt --images wenn >0)")
    p.add_argument("--video-fps", type=int, default=0, help="FPS Override für Video (0 = auto)")
    p.add_argument("--video-blend-mode", choices=["none", "linear", "flow"], default="none", help="Zwischenframe-Glättung: none | linear | flow (optischer Fluss, benötigt OpenCV)")
    p.add_argument("--video-blend-steps", type=int, default=0, help="Anzahl interpolierter Zwischenframes zwischen Originalframes")
    # Interpolation Flags
    p.add_argument("--interp-seed-start", type=int, help="Start-Seed für Interpolation")
    p.add_argument("--interp-seed-end", type=int, help="End-Seed für Interpolation")
    p.add_argument("--interp-frames", type=int, default=0, help="Frames zwischen Seeds (>=2 aktiv)")
    p.add_argument("--interp-slerp", action="store_true", help="Slerp statt linearer Interpolation der Anfangs-Latents")
    # Seed-Cycling / längere Videos ohne Interpolation
    p.add_argument("--seed-cycle", type=int, default=0, help="Erzeuge N Frames durch fortlaufende Seeds (wenn keine Interpolation aktiv)")
    p.add_argument("--seed-step", type=int, default=997, help="Seed Inkrement pro Frame (Primzahl empfohlen)")
    p.add_argument("--latent-jitter", type=float, default=0.0, help="Rauschen (Std) das pro Frame zum Start-Latent addiert wird (nur seed-cycle)")
    # Prompt Morphing
    p.add_argument("--morph-from", help="Start-Prompt für Morphing (aktiv mit --morph-to & --morph-frames)")
    p.add_argument("--morph-to", help="Ziel-Prompt für Morphing")
    p.add_argument("--morph-frames", type=int, default=0, help="Anzahl Frames für Morph (>=2 aktiviert)")
    return p.parse_args()


def infer_defaults(model_id: str, steps: Optional[int], guidance: Optional[float]):
    # Automatische Heuristik
    if steps is None:
        if "turbo" in model_id.lower():
            steps = 2
        else:
            steps = 25
    if guidance is None:
        if "turbo" in model_id.lower():
            guidance = 0.0
        else:
            guidance = 7.5
    return steps, guidance


def load_pipeline(model_id: str, half: bool):
    torch_dtype = torch.float16 if half and torch.cuda.is_available() else None
    if "turbo" in model_id.lower():
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype, variant="fp16" if torch_dtype else None)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    # Attention Slicing nur wenn nicht deaktiviert
    # (wird außerhalb aktiviert, weil args hier nicht verfügbar sind)
    return pipe


def model_stats(pipe) -> dict:
    stats = {}
    try:
        total = 0
        for name, module in [
            ("unet", getattr(pipe, "unet", None)),
            ("vae", getattr(pipe, "vae", None)),
            ("text_encoder", getattr(pipe, "text_encoder", None)),
            ("text_encoder_2", getattr(pipe, "text_encoder_2", None)),
        ]:
            if module is not None:
                params = sum(p.numel() for p in module.parameters())
                total += params
                stats[name] = params
        stats["total"] = total
    except Exception as e:
        stats["error"] = str(e)
    return stats


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    steps, guidance = infer_defaults(args.model, args.steps, args.guidance)

    ensure_outdir(args.outdir)

    # Einheitlicher Basis-Dateiname: <timestamp>-<shortprompt>
    # timestamp: YYYYMMDD-HHMMSS
    ts_run = time.strftime("%Y%m%d-%H%M%S")

    def sanitize_prompt(p: str, max_len: int = 40) -> str:
        import re
        p = p.strip().lower()
        # ersetze umlaute grob
        repl = {"ä":"ae","ö":"oe","ü":"ue","ß":"ss"}
        for k,v in repl.items():
            p = p.replace(k,v)
        # nur a-z0-9 und separator
        p = re.sub(r"[^a-z0-9]+", "-", p)
        p = re.sub(r"-+", "-", p).strip('-')
        if not p:
            p = "prompt"
        if len(p) > max_len:
            p = p[:max_len].rstrip('-')
        return p

    shortprompt = sanitize_prompt(args.prompt)
    base_name = f"{ts_run}-{shortprompt}"
    frame_counter = 0  # global Zähler für alle erzeugten Einzelbilder
    # Per-Run Unterordner anlegen: outputs/<timestamp-shortprompt>
    run_dir = os.path.join(args.outdir, base_name)
    ensure_outdir(run_dir)
    print(f"Run-Verzeichnis: {run_dir}")
    start_time = time.time()

    # Bestimme Modus für Metadaten
    generation_mode = "single"
    # Flags werden später gesetzt sobald klar ist welcher Pfad genutzt wird

    print(f"Lade Pipeline '{args.model}' (steps={steps}, guidance={guidance}) ...")
    pipe = load_pipeline(args.model, args.half)

    if not args.no_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    # Offload Optionen
    if args.seq_offload:
        try:
            pipe.enable_sequential_cpu_offload()
            print("Sequentielles CPU Offload aktiv")
        except Exception as e:
            print("Warnung: seq-offload nicht verfügbar:", e)
    elif args.cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            print("Model CPU Offload aktiv")
        except Exception as e:
            print("Warnung: cpu-offload nicht verfügbar:", e)

    if args.info:
        stats = model_stats(pipe)
        if "error" in stats:
            print("Stat-Fehler:", stats["error"])
        else:
            def fmt(n):
                return f"{n/1e6:.2f}M" if isinstance(n, int) else n
            print("Parameter:")
            for k, v in stats.items():
                if k == 'total':
                    continue
                print(f" - {k:14s}: {fmt(v)}")
            print(f"Gesamt: {fmt(stats['total'])}")
            approx_mem = stats.get('total', 0) * (2 if args.half else 4) / (1024**3)
            print(f"Grober Speicherbedarf (reine Parameter {'FP16' if args.half else 'FP32'}): ~{approx_mem:.2f} GB (ohne Aktivierungen)")
        return

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
        print(f"Seed gesetzt: {args.seed}")

    all_paths: List[str] = []

    # Morph detection
    morph_mode = False
    if args.morph_from and args.morph_to and args.morph_frames and args.morph_frames > 1:
        morph_mode = True
        generation_mode = "morph"
        short_from = sanitize_prompt(args.morph_from)
        short_to = sanitize_prompt(args.morph_to)
        base_name = f"{ts_run}-{short_from}-to-{short_to}"
        run_dir = os.path.join(args.outdir, base_name)
        ensure_outdir(run_dir)
        print(f"Morph aktiv: '{args.morph_from}' -> '{args.morph_to}', frames={args.morph_frames}")
        print(f"Aktualisiertes Run-Verzeichnis: {run_dir}")

    total_images = args.images
    if args.video and args.video_frames > 0:
        total_images = args.video_frames
    if morph_mode:
        total_images = args.morph_frames

    interpolation_mode = (
        args.interp_seed_start is not None and
        args.interp_seed_end is not None and
        args.interp_frames and args.interp_frames > 1
    )
    if not 'morph_mode' in locals():
        morph_mode = False
    if not morph_mode and interpolation_mode:
        total_images = args.interp_frames
    seed_cycle_mode = False
    if not morph_mode and not interpolation_mode and args.seed_cycle > 0:
        total_images = args.seed_cycle
        seed_cycle_mode = True

    # Versuche Batch-Generierung für mehrere Bilder in einem Call (schneller), fallback bei Inkompatibilität
    if not morph_mode and interpolation_mode:
        generation_mode = "interpolation"
        # Latent Interpolation Pfad
        device = pipe.unet.device
        dtype = next(pipe.unet.parameters()).dtype
        height = args.height
        width = args.width
        latent_h = height // 8
        latent_w = width // 8
        in_channels = pipe.unet.config.in_channels if hasattr(pipe.unet, 'config') else 4

        def make_latent(seed:int):
            g = torch.Generator(device=device).manual_seed(seed)
            return torch.randn((1, in_channels, latent_h, latent_w), generator=g, device=device, dtype=dtype)

        lat_start = make_latent(args.interp_seed_start)
        lat_end = make_latent(args.interp_seed_end)

        def slerp(t, v0, v1, eps=1e-7):
            v0_u = v0 / (v0.norm(dim=-1, keepdim=True) + eps)
            v1_u = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
            dot = (v0_u * v1_u).sum(-1, keepdim=True).clamp(-1+1e-6, 1-1e-6)
            omega = torch.arccos(dot)
            so = torch.sin(omega)
            return (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1

        # Flatten last dim for slerp (treat spatial dims as extra dims) -> reshape to (B, N)
        v0_flat = lat_start.view(lat_start.size(0), -1)
        v1_flat = lat_end.view(lat_end.size(0), -1)

        print(f"Interpolation aktiv: seeds {args.interp_seed_start} -> {args.interp_seed_end}, frames={args.interp_frames}, mode={'slerp' if args.interp_slerp else 'linear'}")
        for idx in range(args.interp_frames):
            t = idx / (args.interp_frames - 1)
            if args.interp_slerp:
                blended_flat = slerp(torch.tensor(t, device=device), v0_flat, v1_flat)
            else:
                blended_flat = (1 - t) * v0_flat + t * v1_flat
            blended = blended_flat.view_as(lat_start)
            t0 = time.time()
            result = pipe(
                args.prompt,
                negative_prompt=args.negative or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                latents=blended.clone()
            )
            img: Image.Image = result.images[0]
            frame_counter += 1
            out_path = os.path.join(run_dir, f"{base_name}-{frame_counter:03d}.png")
            img.save(out_path)
            dt = time.time() - t0
            print(f"Frame {idx+1}/{args.interp_frames} t={t:.2f} gespeichert: {out_path} ({dt:.2f}s)")
            all_paths.append(out_path)

    elif not morph_mode and seed_cycle_mode:
        generation_mode = "seed_cycle"
        # Seed-Cycling: pro Frame anderer Seed (sanfte Variation via latent-jitter optional)
        print(f"Seed-Cycle aktiv: frames={total_images}, seed_step={args.seed_step}, latent_jitter={args.latent_jitter}")
        base_seed = args.seed if args.seed is not None else int(time.time()) % 2_147_483_647
        device = pipe.unet.device
        dtype = next(pipe.unet.parameters()).dtype
        latent_h = args.height // 8
        latent_w = args.width // 8
        in_channels = getattr(getattr(pipe, 'unet', None), 'config', None).in_channels if hasattr(getattr(pipe, 'unet', None), 'config') else 4
        base_latent = None
        if args.latent_jitter > 0:
            g0 = torch.Generator(device=device).manual_seed(base_seed)
            base_latent = torch.randn((1, in_channels, latent_h, latent_w), generator=g0, device=device, dtype=dtype)
        for i in range(total_images):
            current_seed = base_seed + i * args.seed_step
            gen_i = torch.Generator(device=device).manual_seed(current_seed)
            latents = None
            if base_latent is not None:
                noise = torch.randn_like(base_latent, generator=gen_i) * args.latent_jitter
                latents = (base_latent + noise).clone()
            t0 = time.time()
            result = pipe(
                args.prompt,
                negative_prompt=args.negative or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=args.height,
                width=args.width,
                generator=gen_i,
                latents=latents
            )
            img: Image.Image = result.images[0]
            frame_counter += 1
            out_path = os.path.join(run_dir, f"{base_name}-{frame_counter:03d}.png")
            img.save(out_path)
            dt = time.time() - t0
            print(f"Frame {i+1}/{total_images} seed={current_seed} gespeichert: {out_path} ({dt:.2f}s)")
            all_paths.append(out_path)
    elif not morph_mode and total_images > 1:
        generation_mode = "batch"
        try:
            t0 = time.time()
            result = pipe(
                args.prompt,
                negative_prompt=args.negative or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=args.height,
                width=args.width,
                num_images_per_prompt=total_images,
                generator=generator
            )
            total_dt = time.time() - t0
            for i, img in enumerate(result.images, 1):
                frame_counter += 1
                out_path = os.path.join(run_dir, f"{base_name}-{frame_counter:03d}.png")
                img.save(out_path)
                print(f"Bild {i}/{total_images} gespeichert: {out_path}")
                all_paths.append(out_path)
            print(f"Batch fertig in {total_dt:.2f}s (Ø {total_dt/total_images:.2f}s/Bild)")
        except TypeError:
            # Fallback einzelner Loop
            for i in range(total_images):
                t0 = time.time()
                result = pipe(
                    args.prompt,
                    negative_prompt=args.negative or None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=args.height,
                    width=args.width,
                    generator=generator
                )
                img: Image.Image = result.images[0]
                frame_counter += 1
                out_path = os.path.join(run_dir, f"{base_name}-{frame_counter:03d}.png")
                img.save(out_path)
                dt = time.time()-t0
                print(f"Bild {i+1}/{total_images} gespeichert: {out_path} ({dt:.2f}s)")
                all_paths.append(out_path)
    elif not morph_mode:
        # Einzelbild wie bisher
        t0 = time.time()
        result = pipe(
            args.prompt,
            negative_prompt=args.negative or None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=args.height,
            width=args.width,
            generator=generator
        )
        img: Image.Image = result.images[0]
        frame_counter += 1
        out_path = os.path.join(run_dir, f"{base_name}-{frame_counter:03d}.png")
        # PNG Metadaten (Prompt etc.)
        try:
            from PIL import PngImagePlugin  # type: ignore
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("prompt", args.prompt)
            if args.negative:
                pnginfo.add_text("negative", args.negative)
            pnginfo.add_text("model", args.model)
            pnginfo.add_text("mode", generation_mode)
            if args.seed is not None:
                pnginfo.add_text("seed", str(args.seed))
            img.save(out_path, pnginfo=pnginfo)
        except Exception:
            img.save(out_path)
        dt = time.time()-t0
        print(f"Bild 1/1 gespeichert: {out_path} ({dt:.2f}s)")
        all_paths.append(out_path)

    # Morphing Branch
    if morph_mode:
        try:
            tokenizer = getattr(pipe, 'tokenizer', None)
            text_encoder = getattr(pipe, 'text_encoder', None)
            if tokenizer is None or text_encoder is None:
                print("[WARN] Morphing nicht unterstützt (kein tokenizer/text_encoder) – deaktiviere Morph")
                morph_mode = False
            else:
                device = pipe.unet.device
                def encode(txt: str):
                    tokens = tokenizer([txt], padding='max_length', truncation=True, max_length=getattr(tokenizer,'model_max_length',77), return_tensors='pt')
                    with torch.no_grad():
                        out = text_encoder(tokens.input_ids.to(device))
                        if hasattr(out, 'last_hidden_state'):
                            emb = out.last_hidden_state
                        elif isinstance(out, (list, tuple)):
                            emb = out[0]
                        else:
                            emb = out
                    return emb
                emb_start = encode(args.morph_from)
                emb_end = encode(args.morph_to)
                neg_emb = None
                if args.negative:
                    try:
                        neg_emb = encode(args.negative)
                    except Exception:
                        pass
                for idx in range(args.morph_frames):
                    t = idx / (args.morph_frames - 1)
                    blended = (1 - t) * emb_start + t * emb_end
                    t0 = time.time()
                    kw = dict(num_inference_steps=steps, guidance_scale=guidance, height=args.height, width=args.width, prompt_embeds=blended)
                    if neg_emb is not None and guidance and guidance > 1.0:
                        kw['negative_prompt_embeds'] = neg_emb
                    result = pipe(**kw)
                    img: Image.Image = result.images[0]
                    frame_counter += 1
                    out_path = os.path.join(run_dir, f"{base_name}-{frame_counter:03d}.png")
                    try:
                        from PIL import PngImagePlugin  # type: ignore
                        pnginfo = PngImagePlugin.PngInfo()
                        pnginfo.add_text("prompt_start", args.morph_from)
                        pnginfo.add_text("prompt_end", args.morph_to)
                        pnginfo.add_text("t", f"{t:.4f}")
                        img.save(out_path, pnginfo=pnginfo)
                    except Exception:
                        img.save(out_path)
                    dt = time.time() - t0
                    print(f"Morph Frame {idx+1}/{args.morph_frames} t={t:.2f} gespeichert: {out_path} ({dt:.2f}s)")
                    all_paths.append(out_path)
        except Exception as e:
            print("[WARN] Morphing fehlgeschlagen:", e)
            morph_mode = False

    print("Fertig. Dateien:")
    for pth in all_paths:
        print(" -", pth)

    # Video-Erstellung
    if args.video and len(all_paths) > 1:
        try:
            import imageio.v2 as imageio
            from PIL import Image
            import numpy as np
            use_fps = args.video_fps if args.video_fps > 0 else (6 if total_images <= 0 else max(2, min(30, total_images // 2 if total_images > 10 else 6)))
            # Einheitlicher Videoname auf Basis des Runs
            video_path = os.path.join(run_dir, f"{base_name}.mp4")
            base_frames = [Image.open(p).convert("RGB") for p in all_paths]

            def pil_to_np(im):
                return np.array(im)

            def np_to_pil(arr):
                return Image.fromarray(arr.astype(np.uint8))

            out_frames = []
            if args.video_blend_mode == "none" or args.video_blend_steps <= 0 or len(base_frames) < 2:
                out_frames = base_frames
            else:
                if args.video_blend_mode == "flow":
                    try:
                        import cv2  # type: ignore
                        have_flow = True
                    except Exception:
                        print("[WARN] OpenCV nicht verfügbar – fallback linear")
                        have_flow = False
                else:
                    have_flow = False

                for i in range(len(base_frames)-1):
                    A = base_frames[i]
                    B = base_frames[i+1]
                    out_frames.append(A)
                    if args.video_blend_steps > 0:
                        A_np = pil_to_np(A)
                        B_np = pil_to_np(B)
                        if have_flow:
                            # Optischer Fluss auf verkleinerte Version für Performance
                            h, w, _ = A_np.shape
                            scale = 512 / max(h, w)
                            if scale < 1:
                                As = cv2.resize(A_np, (int(w*scale), int(h*scale)))
                                Bs = cv2.resize(B_np, (int(w*scale), int(h*scale)))
                            else:
                                As, Bs = A_np, B_np
                            grayA = cv2.cvtColor(As, cv2.COLOR_RGB2GRAY)
                            grayB = cv2.cvtColor(Bs, cv2.COLOR_RGB2GRAY)
                            flow = cv2.calcOpticalFlowFarneback(grayA, grayB, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            fx = cv2.resize(flow[...,0], (w, h))
                            fy = cv2.resize(flow[...,1], (w, h))
                        for s in range(1, args.video_blend_steps+1):
                            t = s / (args.video_blend_steps + 1)
                            if have_flow:
                                # Warp A Richtung B
                                yy, xx = np.mgrid[0:h, 0:w]
                                map_x = (xx + fx * t).astype(np.float32)
                                map_y = (yy + fy * t).astype(np.float32)
                                warpedA = cv2.remap(A_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                                blended = warpedA * (1 - t) + B_np * t
                            else:
                                blended = A_np * (1 - t) + B_np * t
                            out_frames.append(np_to_pil(blended.clip(0,255)))
                out_frames.append(base_frames[-1])

            # In numpy für imageio
            frames_np = [np.array(f) for f in out_frames]
            imageio.mimwrite(video_path, frames_np, fps=use_fps, quality=8)
            print(f"Video gespeichert: {video_path} (FPS={use_fps}, Frames={len(frames_np)}, Blend={args.video_blend_mode}, ExtraFrames={len(frames_np)-len(base_frames)})")
        except Exception as e:
            print("Video-Erstellung fehlgeschlagen:", e)

    # Metadaten-Datei schreiben
    try:
        end_time = time.time()
        meta = {
            "run_id": base_name,
            "timestamp": ts_run,
            "prompt": args.prompt,
            "negative": args.negative or "",
            "model": args.model,
            "mode": generation_mode,
            "width": args.width,
            "height": args.height,
            "steps": steps,
            "guidance": guidance,
            "seed": args.seed,
            "seed_cycle": args.seed_cycle,
            "seed_step": args.seed_step,
            "latent_jitter": args.latent_jitter,
            "interp_seed_start": args.interp_seed_start,
            "interp_seed_end": args.interp_seed_end,
            "interp_frames": args.interp_frames,
            "interp_slerp": args.interp_slerp,
            "video_requested": bool(args.video),
            "video_blend_mode": args.video_blend_mode,
            "video_blend_steps": args.video_blend_steps,
            "video_fps_arg": args.video_fps,
            "images_count": frame_counter,
            "image_files": [os.path.basename(p) for p in all_paths],
            "video_file": f"{base_name}.mp4" if args.video and len(all_paths) > 1 else None,
            "morph_from": args.morph_from,
            "morph_to": args.morph_to,
            "morph_frames": args.morph_frames,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": getattr(__import__('torch'), '__version__', None),
            "diffusers_version": getattr(__import__('diffusers'), '__version__', None),
            "duration_sec": round(end_time - start_time, 3)
        }
        meta_path = os.path.join(run_dir, f"{base_name}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Metadaten gespeichert: {meta_path}")
    except Exception as e:
        print("[WARN] Metadaten nicht geschrieben:", e)


if __name__ == "__main__":
    main()
