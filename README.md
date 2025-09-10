# SDGen – Morph, Interpolation & Video Generator (Deutsch)

Leistungsfähiges, modulares Text‑zu‑Bild / Sequenz / Video Toolkit auf Basis von *Hugging Face diffusers*.

## Inhaltsverzeichnis

1. Überblick & Features  
2. Installation  
3. Schnellstart  
4. Architektur (CLI `generate.py` & Paket `sdgen`)  
5. Modi (single, batch, seed_cycle, interpolation, morph)  
6. Morph – Parameter & Effekte (psychedelisch & Stabilisierung)  
7. Video-Erstellung & Optionen  
8. Performance & Offload  
9. PNG & Run Metadaten  
10. Beispiele (Basic → Fortgeschritten)  
11. Troubleshooting  
12. Erweiterungsideen  
13. Lizenz

---

### Ergänzende Dokumente

- [Detaillierte Dokumentation & Nutzung (DOKU.md)](DOKU.md)
- [Architektur & Diagramme (ARCHITEKTUR.md)](ARCHITEKTUR.md)
- [Web UI Architektur (WEBUI.md)](WEBUI.md)

---

## 1. Überblick & Features

Zentraler Einstieg: `generate.py`.  
Alle Ausgaben landen unter `outputs/<run_id>/` (Bilder, optional MP4, JSON).

Unterstützte Funktionen:

- Schnelle Einzel- & Batch-Generierung
- Seed-Cycle (kontinuierliche Seed-Variation)
- Latent-Interpolation (Seed A → Seed B) linear oder sferisch (Slerp)
- Prompt Morphing (Embeddings) + optional Latent Morph + Effekte
- Psychedelische Morph-Effekte (Farbrotation, Warp, Noise-Puls, Orbit)
- Zeitliche Glättung & Frame-Stabilisierung
- Video-Build mit Blend-Frames (linear / optischer Fluss) + Ziel-Dauer
- Erweiterte Metadaten: JSON pro Run + PNG Text-Chunks pro Bild
- Offload & VRAM-Optimierungen (Half, Slicing, CPU-/Sequential-Offload)

---

## 2. Installation

Voraussetzung: Python 3.10+ (empfohlen 3.12)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (HF Login nötig für bestimmte Modelle):

```bash
huggingface-cli login
```

### Kernabhängigkeiten

- diffusers, torch, transformers, accelerate, safetensors
- Pillow, numpy (<2), imageio + imageio-ffmpeg
- (optional) opencv-python für `--video-blend-mode flow`

---

## 3. Schnellstart

Ultraschnell mit Turbo (minimaler Guidance & Steps):

```bash
python generate.py --prompt "surreal landscape" --model stabilityai/sd-turbo --steps 2 --guidance 0.0 --half
```

Batch:

```bash
python generate.py --prompt "futuristic drone" --images 4 --model stabilityai/sd-turbo --steps 3 --guidance 0.0
```

Morph (einfach):

```bash
python generate.py \
  --prompt placeholder \
  --morph-from "ancient forest" --morph-to "cyberpunk city" \
  --morph-frames 16 --steps 6 --model stabilityai/sd-turbo --guidance 0.0 --half \
  --video --video-blend-mode linear --video-blend-steps 2 --video-target-duration 6
```

---

## 4. Architektur

Dateien:

| Datei / Modul | Zweck |
|---------------|-------|
| `generate.py` | CLI Parsing, Erzeugung `GenerationConfig`, Start der Pipeline |
| `sdgen/config.py` | Dataclass aller Parameter |
| `sdgen/modes.py` | Modus-Erkennung & Orchestrierung, Video + Metadaten |
| `sdgen/morph.py` | Prompt & Latent Morph + Effekte |
| `sdgen/interpolate.py` | Latent Interpolation Seed→Seed |
| `sdgen/seedcycle.py` | Seed-Cycle Generation |
| `sdgen/video.py` | MP4 Builder + Blend Frames |
| `sdgen/utils.py` | Run IDs, Sanitizing, Metadaten, PNG Save |

Modus-Priorität: morph > interpolation > seed_cycle > batch > single.

---

## 5. Modi (Kurzüberblick)

| Modus | Auslöser |
|-------|----------|
| single | Standard (1 Bild) |
| batch | `--images > 1` |
| seed_cycle | `--seed-cycle > 0` |
| interpolation | `--interp-*` Seeds + Frames |
| morph | `--morph-from`, `--morph-to`, `--morph-frames > 1` |
| morph (continuous) | `--morph-prompts >=3 --morph-frames > 1 --morph-continuous` |

### Seed Cycle Flags

`--seed-cycle N --seed-step 997 --latent-jitter 0.05`

### Interpolation Flags

`--interp-seed-start S --interp-seed-end E --interp-frames N [--interp-slerp]`

### Morph Flags (Basis)

`--morph-from A --morph-to B --morph-frames N [--morph-latent --morph-seed-start S --morph-seed-end E --morph-slerp]`

Kontinuierlicher Multi-Prompt Fluss:

`--morph-prompts "p1,p2,p3,..." --morph-frames N --morph-continuous [--morph-latent --morph-slerp]`

---

## 6. Morph – Effekte & Stabilisierung

| Flag | Wirkung |
|------|--------|
| `--morph-ease <linear\|sine\|ease\|ease-in\|ease-out\|quad\|cubic>` | Easing der semantischen Transition |
| `--morph-color-shift` / `--morph-color-intensity f` | Rotierende Farbkanal-Mischung |
| `--morph-noise-pulse f` | Sinusförmige Zusatznoise-Amplitude im Latent |
| `--morph-frame-perturb f` | Leichter horizontaler Warp (sinus) |
| `--morph-effect-curve <center\|linear\|flat\|edges>` | Verteilung der Effektstärke über Zeit |
| `--morph-temporal-blend f` | Mischung mit vorherigem Frame (Stabilität) |
| `--morph-smooth` | Mildes Nachglätten |

Empfohlene „sanft“ Startwerte:

```text
--morph-color-intensity 0.2 --morph-noise-pulse 0.2 --morph-frame-perturb 0.2 \
--morph-temporal-blend 0.35 --morph-effect-curve center --morph-smooth
```

---

## 7. Video

| Flag | Bedeutung |
|------|-----------|
| `--video` | Aktiviert MP4-Erstellung |
| `--video-name name.mp4` | Eigener Dateiname |
| `--video-fps N` | Feste Frames/s (überschreibt Dauer) |
| `--video-target-duration S` | Gewünschte Länge (ermittelt fps) |
| `--video-blend-mode <none\|linear\|flow>` | Zwischenbilder; flow benötigt OpenCV |
| `--video-blend-steps N` | Anzahl Insert-Frames pro Übergang |
| `--video-frames N` | (single/batch) Erzwingt Mindestanzahl Frames (füllt auf) |

Heuristik: Falls `--video-fps` nicht gesetzt und Ziel-Dauer > 0, wird fps dynamisch bestimmt.

---

## 8. Performance & Offload

| Flag | Effekt |
|------|--------|
| `--half` | FP16 (weniger VRAM, schneller) |
| (default) | Attention Slicing aktiv |
| `--no-slicing` | Deaktiviert Slicing (mehr VRAM) |
| `--cpu-offload` | Diffusers CPU Offload |
| `--seq-offload` | Sequentielles Offload (minimaler VRAM, langsamer) |
| `--info` | Nur Modell-Parameter ausgeben, kein Rendern |

`--seq-offload` überschreibt `--cpu-offload` falls gemeinsam gesetzt.

Tipps:

- Turbo Modelle: super wenige Steps (2–6) + guidance 0 für Exploration.
- Erhöhte Steps für Start- & Endframes (Workflow: Morph → wichtige Frames neu rendern).

---

## 9. Metadaten

Jeder Run erzeugt `<run_id>.json` + PNG Text-Chunks (falls unterstützt). Auswahl pro Modus:

| Modus | Wichtige Keys (PNG) |
|-------|---------------------|
| single/batch | prompt, negative, model, mode, seed, steps, guidance |
| seed_cycle | current_seed, seed_step, latent_jitter |
| interpolation | t, slerp, seed_start, seed_end |
| morph | prompt_start, prompt_end, t_raw, t_eased, morph_latent, effect_curve |
| video extend | video_extend=true |
| (global config) | zusätzliche Datei <run_id>-config.json mit vollständiger Konfiguration |
| (summary) | <run_id>-summary.md (Tabellen-Übersicht + Dateien + Roh-Config) |

Auslesen:

```bash
exiftool -a -G -s outputs/<run>/<file>.png
identify -verbose outputs/<run>/<file>.png | grep -i prompt -A2
```

---

## 10. Beispiele

### A) Single Bild

```bash
python generate.py --prompt "studio portrait of a golden retriever" --model runwayml/stable-diffusion-v1-5 --steps 25 --guidance 7.5
```

Konfiguration aus Datei laden (+ Override):

`config.json` Beispiel:

```json
{
  "prompt": "studio portrait of a golden retriever",
  "model": "runwayml/stable-diffusion-v1-5",
  "steps": 25,
  "guidance": 7.5
}
```

Aufruf:

```bash
python generate.py --config config.json --guidance 6.5
```

### B) Batch

```bash
python generate.py --prompt "isometric voxel bakery" --images 6 --model stabilityai/sd-turbo --steps 4 --guidance 0.0
```

### C) Seed Cycle

```bash
python generate.py --prompt "futuristic drone" --seed-cycle 32 --seed-step 997 --latent-jitter 0.05 --model stabilityai/sd-turbo --steps 4 --guidance 0.0 --video --video-blend-mode linear --video-blend-steps 2
```

### D) Interpolation

```bash
python generate.py --prompt "alien desert" \
  --interp-seed-start 111 --interp-seed-end 999 --interp-frames 32 --interp-slerp \
  --model stabilityai/sd-turbo --steps 4 --guidance 0.0 --video --video-target-duration 8
```

### E) Morph Sanft (2-Punkt)

```bash
python generate.py \
  --morph-from "sunrise over desert" --morph-to "stormy ocean at night" \
  --morph-frames 20 --steps 6 --model stabilityai/sd-turbo --guidance 0.0 --half \
  --morph-latent --morph-seed-start 11 --morph-seed-end 222 --morph-slerp --morph-ease ease \
  --morph-color-shift --morph-color-intensity 0.2 \
  --morph-noise-pulse 0.2 --morph-frame-perturb 0.2 \
  --morph-temporal-blend 0.4 --morph-effect-curve center --morph-smooth \
  --video --video-blend-mode linear --video-blend-steps 2 --video-target-duration 8 --prompt placeholder
```

### F) Morph Kreativ (2-Punkt)

```bash
python generate.py \
  --morph-from "ancient crystalline library floating in space" \
  --morph-to "bioluminescent coral cathedral under the deep sea" \
  --morph-frames 28 --steps 6 --model stabilityai/sd-turbo --guidance 0.0 --half \
  --morph-latent --morph-seed-start 321 --morph-seed-end 9876 --morph-slerp --morph-ease sine \
  --morph-color-shift --morph-color-intensity 0.45 \
  --morph-noise-pulse 0.35 --morph-frame-perturb 0.35 \
  --morph-temporal-blend 0.3 --morph-effect-curve center --morph-smooth \
  --video --video-blend-mode linear --video-blend-steps 3 --video-target-duration 12 --prompt placeholder
```

### G) Morph Sequenz (mehrere Stufen)

```bash
python generate.py \
  --morph-prompts "ancient forest, misty ruins, neon alley, cyberpunk skyline" \
  --morph-frames 40 --steps 6 --model stabilityai/sd-turbo --guidance 0.0 --half \
  --morph-latent --morph-seed-start 1111 --morph-seed-end 9999 --morph-slerp --morph-ease ease \
  --morph-color-shift --morph-color-intensity 0.25 \
  --morph-noise-pulse 0.2 --morph-frame-perturb 0.2 \
  --morph-temporal-blend 0.35 --morph-effect-curve center --morph-smooth \
  --video --video-blend-mode linear --video-blend-steps 2 --video-target-duration 10 --prompt placeholder
```

---

## 11. Troubleshooting

| Problem | Ursache | Lösung |
|---------|---------|-------|
| Kein Video | <2 Frames / Fehler beim Build | Prüfen ob mehrere Bilder entstanden, Logs ansehen |
| Flimmern | Zu wenig temporale Bindung | `--morph-temporal-blend` erhöhen, Noise/Frame Perturb senken |
| Farbchaos | Zu hohe Farbintensität | `--morph-color-intensity <0.3` |
| Unscharfe Endframes | Steps zu gering | Steps erhöhen oder Guidance moderat anheben |
| Flow Blend ignoriert | OpenCV fehlt | `pip install opencv-python` |
| PNG-Metadaten fehlen | Pillow PNGInfo Problem | JSON nutzen oder Pillow updaten |

---

## 12. Erweiterungsideen

- GIF / WebM Export
- Abschaltbare PNG-Metadaten (`--no-png-meta`)
- Ping-Pong Schleifen für Morph/Interpolation
- Weitere Effektkurven & adaptives Rauschmanagement

---

## 13. Lizenz

Quellcode: Public Domain (CC0).  
Modelle: jeweilige Modell-Lizenzen beachten.

---

Viel Spaß beim Explorieren und Morphen!
