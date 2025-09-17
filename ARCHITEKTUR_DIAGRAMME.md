# Architektur Diagramme (Ergänzung)

Diese Datei ergänzt `ARCHITEKTUR.md` und `WEBUI.md` um zusätzliche fokussierte Visualisierungen (C4-inspiriert).

## 1. System Context

```mermaid
flowchart LR
  User([Creator / CLI User]) --> CLI[generate.py]
  User --> Web[Gradio Web UI]
  CLI --> Core[SDGen Core Module]
  Web --> Core
  Core --> HF[(Hugging Face Hub)]
  Core --> FS[(Dateisystem / outputs)]
  FS --> Viewer[preview_viewer.py]
  subgraph External Services
    HF
  end
```

## 2. Container / Komponenten Überblick

```mermaid
flowchart TB
  subgraph CLI Layer
    GCLI[generate.py]
  end
  subgraph UI Layer
    WUI[webui.py]
  end
  subgraph Core
    CFG[config.py\nGenerationConfig]
    MOD[modes.py\nMode Dispatcher]
    MOR[morph.py]
    INT[interpolate.py]
    SC[seedcycle.py]
    VID[video.py]
    PIPE[pipeline.py]
    UTL[utils.py]
  end
  GCLI --> MOD
  WUI --> MOD
  MOD --> MOR
  MOD --> INT
  MOD --> SC
  MOD --> VID
  MOD --> PIPE
  PIPE --> HF[(HF Models)]
  MOR --> PIPE
  INT --> PIPE
  SC --> PIPE
  VID --> FS[(outputs/*)]
  UTL --> FS
  MOD --> FS
```

## 3. Datenfluss Konfiguration → Bilder

```mermaid
sequenceDiagram
  participant U as User
  participant G as generate.py
  participant C as config.py
  participant P as pipeline.py
  participant M as modes.py
  participant X as Generator (morph/interp/...)
  participant F as FileSystem
  U->>G: Flags / --config
  G->>C: Merge + Validation
  G->>P: build_pipeline()
  P-->>G: Pipeline
  G->>M: run_generation(cfg, pipeline)
  M->>X: dispatch(cfg)
  loop Frames
    X->>P: inference step
    P-->>X: latents -> image
    X->>F: write image + meta
  end
  M-->>G: Result Pfade
```

## 4. Morph Detail (Embedding & Latent)

```mermaid
flowchart LR
  A[Prompts Liste] --> ENC[Text Encoder]
  ENC --> EMB[Prompt Embeddings]
  EMB -->|Frame t Auswahl + Easing| BLEND[Embedding Interpolation]
  BLEND --> UNet[UNet Denoising]
  subgraph Optional Latent Morph
    LAT0[Latent Seed A] --> LBLEND[Latent Interp (linear/slerp)]
    LAT1[Latent Seed B] --> LBLEND
  end
  LBLEND --> UNet
  UNet --> IMG[Frame PNG]
  IMG --> POST[Effekte / Temporal Blend]
  POST --> OUT[(Run Ordner)]
```

## 5. Video Build Pipeline

```mermaid
flowchart TB
  FR[Frames] --> BM{Blend Mode}
  BM -->|none| MP4[Encode MP4]
  BM -->|linear| LIN[Generate Inbetween Frames]
  BM -->|flow| FLO[Optical Flow Frames]
  LIN --> MP4
  FLO --> MP4
  MP4 --> TD{Target Duration?}
  TD -->|yes & fps=0| CALC[Recalc FPS]
  TD -->|no| DONE((Done))
  CALC --> REMUX[Remux]
  REMUX --> DONE
```

## 6. Threading in Web UI

```mermaid
sequenceDiagram
  participant U as User
  participant W as webui.py
  participant T as Worker Thread
  participant PC as Pipeline Cache
  participant P as build_pipeline
  participant G as run_generation
  U->>W: Klick Generate
  W->>T: start thread(cfg_dict)
  T->>PC: lookup(model, half)
  alt Miss
    T->>P: build_pipeline()
    P-->>T: pipeline
    T->>PC: store
  end
  T->>G: run_generation(cfg, pipeline)
  G->>G: write frames
  G-->>T: Pfade
  T-->>W: update APP_STATE
```

## 7. Zustandsautomat (erweitert)

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> Generating: Generate
  Generating --> Completed: success
  Generating --> Error: exception
  Completed --> Generating: new run
  Error --> Generating: retry
  Completed --> Idle: clear (geplant)
  Error --> Idle: reset (geplant)
```

## 8. Dateistruktur (vereinfacht)

```mermaid
flowchart LR
  ROOT[difffusers-try/] --> GEN[generate.py]
  ROOT --> WEB[webui.py]
  ROOT --> CFG[config_rainbow.json]
  ROOT --> OUT[outputs/]
  OUT --> RUN1[2025-09-15_123456_runid/]
  RUN1 --> IMG1[0001.png]
  RUN1 --> META1[0001.json]
  RUN1 --> VID1[runid.mp4]
```

## 9. Erweiterungspunkte

```mermaid
flowchart TB
  CORE_EXT[Neue Modi] --> MOD
  UI_EXT[Progress / Cancel] --> WUI
  PERF[Flash Attention / xFormers] --> PIPE
  CACHE[Pipeline Eviction] --> WUI
  META[Erweiterte Summary Reports] --> UTL
  DIST[Mehr-GPU / Remote Worker] --> MOD
```

---
Letzte Aktualisierung: 2025-09-15
