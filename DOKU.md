# Morph & Video Generator – Deutsche Dokumentation

Diese Dokumentation beschreibt das modulare Skript `generate.py` und das Paket `sdgen`, mit dem fortgeschrittene Bildserien, Morph-Sequenzen und Videos aus Text-Prompts erzeugt werden.

## Inhalt

- Überblick
- Installation & Voraussetzungen
- Grundlegende Nutzung
- Verzeichnisstruktur & Run-Ordner
- Modi (single, batch, seed_cycle, interpolation, morph)
- Morphing: Parameter & Effekte
- Video-Erstellung
- Beispiele
- Performance & Tipps
- Troubleshooting

---

## Überblick

`generate.py` nutzt das interne Paket `sdgen`, um verschiedene Generations-Modi auf Basis von *Stable Diffusion* (z.B. `stabilityai/sd-turbo`) auszuführen. Es unterstützt:

- Einzel- und Batch-Generierung
- Seed-Cycle (fortlaufende Variation über Seeds)
- Latent-Interpolation zwischen Seeds
- Prompt-Morphing (Embedding + optional Latent-Slerp)
- Psychedelische Effekte (Farbverschiebung, Noise-Puls, Pixel-Warp)
- Glättung & zeitliche Stabilisierung
- Video-Zusammenbau inkl. Blend-Frames und Ziel-Dauer
- Metadaten pro Run (JSON)
- PNG Text-Metadaten pro Bild (Prompt / Modus / Parameter)



Alle Ausgaben landen standardmäßig unter `outputs/<run_id>/`.

---

## Installation & Voraussetzungen

Python 3.10+ (empfohlen 3.12) und eine funktionierende `pip` Umgebung.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: Falls ein Modell Login verlangt:

```bash
huggingface-cli login
```

### Abhängigkeiten (Kern)

- diffusers
- torch
- transformers
- accelerate
- safetensors
- pillow, numpy, imageio[ffmpeg]
- (optional) opencv-python für `--video-blend-mode flow`

---

## Grundlegende Nutzung

Ein einfaches einzelnes Bild (Turbo, sehr wenige Steps, guidance 0):

```bash
python generate.py --prompt "surreal landscape" --model stabilityai/sd-turbo --steps 2 --guidance 0.0 --half
```

Mehrere Bilder (Batch):

```bash
python generate.py --prompt "futuristic drone" --images 4 --model stabilityai/sd-turbo --steps 3 --guidance 0.0
```

---

## Verzeichnisstruktur & Run-Ordner

Jeder Lauf erzeugt einen eindeutigen `run_id` basierend auf Zeitstempel + gekürztem Prompt. Beispiel:

```text
outputs/20250910-120222-placeholder/
  20250910-120222-placeholder-001.png
  ...
  20250910-120222-placeholder.json
  20250910-120222-placeholder.mp4 (falls Video)
```

Die JSON-Datei enthält Parameter & Dateiliste.

---

## Modi

Der Modus wird automatisch bestimmt (Priorität: morph > interpolation > seed_cycle > batch > single).

### 1. Single / Batch

Parameter: `--images N`, `--seed`, `--height`, `--width`, `--steps`, `--guidance`.

### 2. Seed Cycle

Erzeugt Sequenz unterschiedlicher Seeds.

```bash
--seed-cycle 40 --seed-step 997 --latent-jitter 0.05
```

- `seed_cycle`: Anzahl Frames
- `seed_step`: Inkrement pro Frame
- `latent_jitter`: Rauschen zusätzlich (kleine Werte 0.0–0.1)

### 3. Latent Interpolation

Verbindet zwei Seeds im Latent-Raum über Zwischenzustände.

| Flag | Beschreibung |
|------|--------------|
| `--interp-seed-start S` | Start-Seed |
| `--interp-seed-end E` | Ziel-Seed |
| `--interp-frames N` | Anzahl Frames (>=2) |
| `--interp-slerp` | Sphärische statt linearer Interpolation |

Beispiel:

```bash
python generate.py --prompt "ancient sculpture" \
  --interp-seed-start 111 --interp-seed-end 999 --interp-frames 24 --interp-slerp \
  --model stabilityai/sd-turbo --steps 4 --guidance 0.0 --half \
  --video --video-blend-mode linear --video-blend-steps 2
```

### 4. Morph

Interpoliert Prompt-Embeddings (und optional Latents) zwischen zwei Texten.

```bash
--morph-from "start prompt" --morph-to "end prompt" --morph-frames 24 \
--morph-latent --morph-seed-start 11 --morph-seed-end 222 --morph-slerp
```

### 5. Morph Psychedelic & Stabilisierung

| Parameter | Wirkung |
|-----------|---------|
| `--morph-color-shift` | Kanalrotation für Farbfluss |
| `--morph-color-intensity f` | Stärke der Farbverschiebung (0–1) |
| `--morph-noise-pulse f` | Sinusförmige zusätzliche Latent-Rausch-Amplitude |
| `--morph-frame-perturb f` | Sinus-Pixel-Warp horizontal |
| `--morph-ease <linear\|sine\|ease\|...>` | Easing-Kurve für semantischen Übergang |
| `--morph-temporal-blend f` | Mischen mit vorherigem Frame (0–1) für Stabilität |
| `--morph-effect-curve <center\|linear\|flat\|edges>` | Verlauf der Effektstärke über Zeit |
| `--morph-smooth` | Mildes Nachglätten (reduziert Flimmern) |

Empfohlene Startwerte für „sanft“:

```bash
--morph-color-intensity 0.2 --morph-noise-pulse 0.2 --morph-frame-perturb 0.2 --morph-temporal-blend 0.35 --morph-effect-curve center --morph-smooth
```

---

## Video-Erstellung

Aktivierung durch `--video`.

Wichtige Parameter:

| Flag | Bedeutung |
|------|-----------|
| `--video-name name.mp4` | Optional eigener Dateiname |
| `--video-fps N` | Fixes fps (überschreibt Dauerheuristik) |
| `--video-target-duration 12` | Ziel-Länge in Sekunden (berechnet fps wenn `--video-fps` 0) |
| `--video-blend-mode <none\|linear\|flow>` | Zusätzliche Zwischenbilder (flow benötigt OpenCV) |
| `--video-blend-steps N` | Anzahl Blendframes zwischen zwei Keyframes |
| `--video-frames N` | (nur single/batch) Erzwingt Gesamtzahl Frames; füllt per Zusatzbilder auf |

Tipp: Höhere Blend-Steps glätten, erhöhen aber Rechenzeit / Dateigröße.

---

## Beispiele

### Sanftes Morph (stabil)

```bash
python generate.py \
  --morph-from "sunrise over desert" \
  --morph-to "stormy ocean at night" \
  --morph-frames 20 --steps 6 --model stabilityai/sd-turbo --guidance 0.0 --half \
  --morph-latent --morph-seed-start 11 --morph-seed-end 222 --morph-slerp --morph-ease ease \
  --morph-color-shift --morph-color-intensity 0.2 \
  --morph-noise-pulse 0.2 --morph-frame-perturb 0.2 \
  --morph-temporal-blend 0.4 --morph-effect-curve center --morph-smooth \
  --video --video-blend-mode linear --video-blend-steps 2 --video-target-duration 8 --prompt placeholder
```

### Kreativeres Psychedelisches Morph

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

---

## Performance & Tipps

- Turbo + niedrige Steps (2–6) für schnelle Morph-Prototypen.
- Etwas höhere Steps (8–12) an Anfang/Ende liefern schärfere Start-/Zielbilder; Idee: zweistufiger Workflow (erst Morph generieren, danach Einzelbilder neu rendern mit mehr Steps und ersetzen).
- `--morph-temporal-blend` > 0.5 kann Details verwischen – moderat bleiben.
- Farb- und Warp-Effekte vorsichtig dosieren (≤0.5) für Lesbarkeit.

### Offload / Ressourcen Flags

| Flag | Wirkung |
|------|--------|
| `--half` | FP16 falls möglich (weniger VRAM, schneller) |
| (Standard) | Attention Slicing aktiv (VRAM sparen) |
| `--no-slicing` | Deaktiviert Slicing (etwas schneller, mehr VRAM) |
| `--cpu-offload` | Automatisches CPU Offload (diffusers) |
| `--seq-offload` | Sequentielles Offload (max VRAM-Ersparnis, langsamer) |
| `--info` | Nur Modell-Parameter zählen, keine Bilder |

`--seq-offload` überschreibt `--cpu-offload` falls beides angegeben.

---

## Troubleshooting

| Problem | Ursache | Lösung |
|---------|---------|-------|
| Video fehlt | Fehler beim Video-Build / keine >1 Frames | Prüfen ob mehrere Bilder erzeugt wurden, Logs ansehen |
| Flimmern / Flicker | Zu wenig zeitliche Korrelation | `--morph-temporal-blend` erhöhen, `--morph-noise-pulse` senken |
| Farben zu extrem | Hohe `--morph-color-intensity` | Wert reduzieren (<0.3) |
| Unscharfe Endframes | Zu wenige Steps / guidance 0 | Steps erhöhen oder geringe guidance (>1) testen |
| Flow Blend ignoriert | OpenCV nicht installiert | `pip install opencv-python` |
| PNG-Metadaten fehlen | Schreibfehler / PIL ohne PNGInfo | JSON-Datei nutzen oder Pillow aktualisieren |

---

## Geplante Erweiterungen (Ideen)

- Ping-Pong Loop Flag
- Kaleidoskop / Spiegel-Symmetrie
- GIF / WebM Export
- Adaptive Effekt-Attenuation bei hoher Detaildichte
- GIF / WebM Export
- Abschaltbare PNG-Metadaten (`--no-png-meta`)
- Ping-Pong Schleife für Morph / Interpolation

---

## PNG Metadaten

Jedes PNG enthält – soweit möglich – Text-Chunks mit Parametern.

| Modus | Schlüssel (Auswahl) |
|-------|---------------------|
| single/batch | prompt, negative, model, mode, seed, steps, guidance |
| seed_cycle | prompt, mode, current_seed, seed_step, latent_jitter |
| interpolation | prompt, mode, t, slerp, seed_start, seed_end |
| morph | prompt_start, prompt_end, t_raw, t_eased, morph_latent, morph_slerp, effect_curve |
| video extend | video_extend=true zusätzlich |

Anzeigen:

```bash
exiftool -a -G -s outputs/<run>/<file>.png
identify -verbose outputs/<run>/<file>.png | grep -i prompt -A2
```

Fällt das Schreiben aus, bleibt die PNG nutzbar; Run-JSON enthält Gesamtdaten.

---

## Lizenz

Dieses Dokument und die Skripte: Public Domain (CC0). Modelle unterliegen ihren eigenen Lizenzen.
