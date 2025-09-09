# Diffusers Beispiel (Deutsch)

Ein minimaler Einstieg zur Nutzung von *Hugging Face diffusers* für Text-zu-Bild.

## Installation

Python 3.10+ empfohlen.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

(Optional) Bei Modellen, die Auth benötigen:

```bash
huggingface-cli login
```

## Nutzung

Schnelles Turbo-Modell:

```bash
python generate.py --prompt "ein fotorealistischer roter Oldtimer vor Bergpanorama" --model stabilityai/sd-turbo --steps 2 --guidance 0.0
```

Klassisches Stable Diffusion 1.5:

```bash
python generate.py --prompt "studio lighting portrait of a golden retriever" --model runwayml/stable-diffusion-v1-5 --steps 25 --guidance 7.5
```

Mehr Optionen:

```bash
python generate.py \
  --prompt "cinematic wide angle shot of a neon cyberpunk street, rain" \
  --negative "low quality, blurry" \
  --model runwayml/stable-diffusion-v1-5 \
  --height 512 --width 768 \
  --steps 30 --guidance 8 \
  --images 3 \
  --seed 42 \
  --half
```

Erzeugte Bilder liegen in `outputs/`.

## Tipps

- Turbo: sehr schnell, geringe Steps (1–4), guidance meist 0.
- Klassisch: mehr Steps 20–35, guidance 6–9 für stärkere Prompt-Treue.
- Für reproduzierbare Ergebnisse `--seed` setzen.
- Größere Auflösungen brauchen mehr VRAM; starte mit 512x512.

## Lizenz

Dieses Beispiel-Skript ist Public Domain (CC0). Nutzung auf eigene Verantwortung. Beachte die Lizenzbedingungen der verwendeten Modelle.
