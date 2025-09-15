# Standalone Video Exporter

Ein eigenständiger, kommandozeilenbasierter Video-Exporter mit FFmpeg-basierter Frame-Interpolation.

## Features

- ✅ **FFmpeg-basierte Interpolation** - Nutzt FFmpeg's `minterpolate` Filter für flüssige Bewegungen
- ✅ **CLI-Interface** - Einfache Bedienung über Kommandozeile
- ✅ **Multiple Qualitätspresets** - YouTube, High, Medium, Low
- ✅ **Upscaling-Support** - 2x, 4x oder beliebige Faktoren
- ✅ **Automatische Codec-Auswahl** - libx264, libx265, etc.
- ✅ **Progress-Tracking** - Echtzeit-Fortschrittsanzeige
- ✅ **Robuste Fehlerbehandlung** - Detaillierte Fehlermeldungen

## Installation

Keine Installation erforderlich! Benötigt nur:
- Python 3.6+
- FFmpeg (automatisch erkannt in Projektordner oder System-PATH)

## Verwendung

### Grundlegende Syntax
```bash
python video_exporter.py <input_folder> <output.mp4> [optionen]
```

### Beispiele

#### 1. Einfaches Video (30 FPS, mittlere Qualität)
```bash
python video_exporter.py input/ output.mp4
```

#### 2. Hochqualitätsvideo mit Upscaling
```bash
python video_exporter.py input/ output.mp4 --fps 60 --quality high --upscale 2x
```

#### 3. Mit Frame-Interpolation (30→60 FPS)
```bash
python video_exporter.py input/ output.mp4 --fps 30 --interpolation-fps 60 --interpolation-mode mci
```

#### 4. YouTube-optimiert mit H.265
```bash
python video_exporter.py input/ output.mp4 --codec libx265 --quality youtube --fps 60
```

#### 5. Maximale Qualität mit 4x Upscaling
```bash
python video_exporter.py input/ output.mp4 --fps 30 --interpolation-fps 120 --upscale 4x --quality high
```

## Parameter-Referenz

### Erforderliche Parameter
- `input_folder` - Ordner mit Eingabebildern (PNG, JPG, etc.)
- `output` - Ausgabe-Video-Datei (.mp4, .mov, etc.)

### Video-Einstellungen
- `--fps` - Basis-Framerate (Standard: 30)
- `--quality` - Qualitätsvoreinstellung:
  - `youtube` - Optimiert für YouTube (CRF 18)
  - `high` - Hohe Qualität (CRF 20)
  - `medium` - Mittlere Qualität (CRF 23)
  - `low` - Niedrige Qualität (CRF 28)
- `--upscale` - Skalierungsfaktor (z.B. `2x`, `1.5x`)
- `--codec` - Video-Codec (Standard: libx264)

### Interpolations-Einstellungen
- `--interpolation-fps` - Ziel-FPS für Interpolation
- `--interpolation-mode` - Interpolationsmodus:
  - `mci` - Motion Compensated Interpolation (beste Qualität)
  - `blend` - Einfaches Blending (mittel)
  - `dup` - Frame-Duplikation (schnellste)

### Sonstige Optionen
- `--verbose, -v` - Detaillierte Ausgabe
- `--ffmpeg-path` - Benutzerdefinierter FFmpeg-Pfad

## Interpolationsmodi

### MCI (Motion Compensated Interpolation)
- **Beste Qualität** für bewegte Objekte
- Erkennt Bewegungsrichtungen und erstellt echte Zwischenframes
- Langsamer, aber sehr glatte Ergebnisse

### Blend
- **Mittlere Qualität** durch Überblendung
- Schneller als MCI
- Gut für langsame Bewegungen

### Dup (Duplication)
- **Schnellste** Option
- Dupliziert einfach Frames
- Keine echte Interpolation, aber höhere FPS

## Ausgabebeispiele

```
🎬 Standalone Video Exporter v1.0
========================================
🎬 Creating video from 260 images
📁 Input: inputs/
🎥 Output: output.mp4
⚙️ Settings: 30fps, high quality, 2x scale, libx264 codec
🔄 Interpolation: 30fps → 60fps using mci
📋 Prepared 260/260 images...
🎬 FFmpeg command: ffmpeg -y -framerate 30 -i temp/frame_%06d.png -vf minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=iw*2:ih*2:flags=lanczos -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p -movflags +faststart output.mp4
🎬 Starting video creation...
✅ Video created successfully in 45.2s!
📄 Output: output.mp4
📊 File size: 125.6 MB
🎉 Export completed successfully!
```

## FFmpeg-Integration

Das Tool nutzt FFmpeg's erweiterte Filter für professionelle Ergebnisse:

- **minterpolate** - Hochwertige Frame-Interpolation
- **scale** - Lanczos-Upscaling für scharfe Ergebnisse
- **Optimierte Presets** - Balanciert Qualität und Dateigröße
- **Fast-Start** - Optimiert für Web-Streaming

## Fehlerbehebung

### FFmpeg nicht gefunden
```bash
# Option 1: FFmpeg in Projekt-Ordner platzieren
ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe

# Option 2: Benutzerdefinierten Pfad angeben
python video_exporter.py input/ output.mp4 --ffmpeg-path "C:/ffmpeg/bin/ffmpeg.exe"

# Option 3: FFmpeg in System-PATH installieren
```

### Keine Bilder gefunden
- Unterstützte Formate: PNG, JPG, JPEG, BMP, TIFF, TGA
- Bilder müssen direkt im angegebenen Ordner liegen
- Verwenden Sie `--verbose` für detaillierte Informationen

### Interpolation zu langsam
- Verwenden Sie `--interpolation-mode blend` oder `dup`
- Reduzieren Sie die Ziel-FPS
- Verwenden Sie kleinere Upscaling-Faktoren

## Performance-Tipps

1. **Für maximale Geschwindigkeit:**
   ```bash
   python video_exporter.py input/ output.mp4 --quality low --interpolation-mode dup
   ```

2. **Für beste Qualität:**
   ```bash
   python video_exporter.py input/ output.mp4 --quality youtube --interpolation-mode mci
   ```

3. **Für große Projekte:**
   - Verwenden Sie SSD-Speicher für Temp-Dateien
   - Schließen Sie andere Anwendungen
   - Nutzen Sie `--verbose` zur Fortschrittsverfolgung
