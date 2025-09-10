#!/usr/bin/env bash
set -euo pipefail

# start.sh - Convenience Wrapper für generate.py
# Erstellt/benutzt virtuelle Umgebung (.venv312), installiert Abhängigkeiten falls nötig
# und führt danach die Bildgenerierung aus.
#
# Nutzung:
#   ./start.sh "<PROMPT TEXT>" [weitere generate.py Argumente]
# Beispiele:
#   ./start.sh "fotorealistischer roter Oldtimer vor Bergpanorama" --model stabilityai/sd-turbo --steps 2 --guidance 0.0 --images 3
#   ./start.sh "studio lighting portrait of a golden retriever" --model runwayml/stable-diffusion-v1-5 --steps 25 --guidance 7.5
#   PROMPT="epische Landschaft" ./start.sh --model stabilityai/sd-turbo --images 4
#
# Optional über Umgebungsvariablen:
#   PROMPT   Prompt Text falls kein erster Parameter
#   PYTHON12 Pfad zu python3.12 (Default: autodetect python3.12 / python3)
#
# Hinweise:
#   - Beim ersten Lauf kann das Laden großer Modellgewichte mehrere Minuten dauern.
#   - Abbrechen mit Ctrl+C.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

REQ_FILE="requirements.txt"
VENV_DIR=".venv312"
PYTHON_BIN="${PYTHON12:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[FEHLER] Kein python3 gefunden." >&2
    exit 1
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Erzeuge virtuelle Umgebung ($VENV_DIR) mit $PYTHON_BIN ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Aktivieren
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Prüfe ob requirements installiert (heuristisch durch Vorhandensein von torch)
if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "[INFO] Installiere Abhängigkeiten ..."
  pip install --upgrade pip >/dev/null
  pip install -r "$REQ_FILE"
fi

# Prompt sammeln
if [[ $# -gt 0 && ! $1 == --* ]]; then
  PROMPT_INPUT="$1"; shift
elif [[ -n "${PROMPT:-}" ]]; then
  PROMPT_INPUT="$PROMPT"
else
  echo "Nutzung: $0 \"<PROMPT>\" [generate.py Optionen]" >&2
  exit 1
fi

# Standard Flags falls nicht gesetzt: turbo defaults
EXTRA_ARGS=()

# Wenn keine Steps angegeben -> nichts tun, generate.py setzt heuristik
# Falls Benutzer keine guidance angegeben hat -> generate.py heuristik

SCRIPT="generate.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "[FEHLER] $SCRIPT nicht gefunden." >&2
  exit 1
fi

echo "[INFO] Starte Generierung: '$PROMPT_INPUT' $*"
python "$SCRIPT" --prompt "$PROMPT_INPUT" "$@"

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[INFO] Fertig. Ausgabedateien unter outputs/"
else
  echo "[WARN] generate.py beendet mit Code $EXIT_CODE" >&2
fi
exit $EXIT_CODE
