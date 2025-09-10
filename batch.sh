#!/usr/bin/env bash
set -euo pipefail
# batch.sh – Generiert mehrere Motive über ./start.sh
# Anpassbare Standardparameter unten.

MODEL="stabilityai/sd-turbo"
STEPS=2
GUIDANCE=0.0
IMAGES=1          # Anzahl Bilder pro Prompt
EXTRA_FLAGS=(--half)  # Weitere Flags für generate.py (z.B. --cpu-offload)

# Ausgabeverzeichnis (wird in generate.py erneut sichergestellt)
OUTDIR="outputs"
MANIFEST="${OUTDIR}/batch_manifest_$(date +%Y%m%d_%H%M%S).csv"

PROMPTS=(
  "Ein Astronaut, der in einem surrealen Wald aus leuchtenden Pilzen spazieren geht, während ein freundlicher Roboterhase ihn begleitet."
  "Eine dampfbetriebene Teekanne, die durch die Wolken fliegt, gezogen von winzigen Kolibris."
  "Eine futuristische Stadt, die auf riesigen Lotusblättern schwimmt, mit Wasserfällen, die von den Wolken herabfallen."
  "Ein neugieriger Bär, der in einem uralten Bibliotheksraum in den Bergen sitzt und Bücher liest, beleuchtet von warmem Kerzenlicht."
  "Eine alte mechanische Eule, die in einem verlassenen Uhrenturm lebt und die Sterne mit einem Teleskop aus Zahnrädern beobachtet."
  "Ein gigantisches Walross, das eine goldene Krone trägt und auf einem Eisberg aus Edelsteinen ruht, umgeben von einem Meer aus flüssigem Sternenlicht."
  "Ein flauschiges, dreiäugiges Wesen, das in einem Baumhaus aus Zuckerwatte wohnt und mit Regenbögen malt."
  "Ein fliegendes Piratenschiff, das durch einen Sturmwolken-Ozean segelt, mit einem Mast, der aus einem riesigen, verzauberten Baum wächst."
  "Ein alter Zauberer, der in einer Höhle aus leuchtenden Kristallen sitzt und einen winzigen Drachenbabys beibringt, wie man Feuer spuckt."
  "Ein Straßenmusiker, der eine Harfe aus gefrorenem Licht spielt, während Schneeflocken in Form von Noten auf ihn herabfallen."
)

if [[ ! -x ./start.sh ]]; then
  echo "[FEHLER] start.sh nicht gefunden oder nicht ausführbar" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

echo "prompt;model;steps;guidance;images;timestamp" > "$MANIFEST"

# Funktion für Zeitformat
now() { date +%Y-%m-%dT%H:%M:%S; }

# Hauptschleife
idx=0
for prompt in "${PROMPTS[@]}"; do
  idx=$((idx+1))
  echo "[INFO] (${idx}/${#PROMPTS[@]}) Generiere: $prompt"
  # Aufruf von start.sh – leitet an generate.py weiter
  ./start.sh "$prompt" \
    --model "$MODEL" \
    --steps "$STEPS" \
    --guidance "$GUIDANCE" \
    --images "$IMAGES" \
    --outdir "$OUTDIR" \
    "${EXTRA_FLAGS[@]}"

  echo "${prompt//;/,};$MODEL;$STEPS;$GUIDANCE;$IMAGES;$(now)" >> "$MANIFEST"
  echo "[OK] Fertig: ${idx}/${#PROMPTS[@]}"
  echo
done

echo "[DONE] Alle Prompts abgeschlossen. Manifest: $MANIFEST"
