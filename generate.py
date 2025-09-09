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

    # Versuche Batch-Generierung für mehrere Bilder in einem Call (schneller), fallback bei Inkompatibilität
    if args.images > 1:
        try:
            t0 = time.time()
            result = pipe(
                args.prompt,
                negative_prompt=args.negative or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=args.height,
                width=args.width,
                num_images_per_prompt=args.images,
                generator=generator
            )
            total_dt = time.time() - t0
            for i, img in enumerate(result.images, 1):
                ts = int(time.time()*1000)
                out_path = os.path.join(args.outdir, f"gen_{ts}_{i}.png")
                img.save(out_path)
                print(f"Bild {i}/{args.images} gespeichert: {out_path}")
                all_paths.append(out_path)
            print(f"Batch fertig in {total_dt:.2f}s (Ø {total_dt/args.images:.2f}s/Bild)")
        except TypeError:
            # Fallback einzelner Loop
            for i in range(args.images):
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
                ts = int(time.time()*1000)
                out_path = os.path.join(args.outdir, f"gen_{ts}_{i+1}.png")
                img.save(out_path)
                dt = time.time()-t0
                print(f"Bild {i+1}/{args.images} gespeichert: {out_path} ({dt:.2f}s)")
                all_paths.append(out_path)
    else:
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
        ts = int(time.time()*1000)
        out_path = os.path.join(args.outdir, f"gen_{ts}_1.png")
        img.save(out_path)
        dt = time.time()-t0
        print(f"Bild 1/1 gespeichert: {out_path} ({dt:.2f}s)")
        all_paths.append(out_path)

    print("Fertig. Dateien:")
    for pth in all_paths:
        print(" -", pth)


if __name__ == "__main__":
    main()
