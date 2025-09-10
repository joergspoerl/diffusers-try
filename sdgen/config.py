from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationConfig:
    prompt: str
    negative: str = ""
    model: str = "stabilityai/sd-turbo"
    height: int = 512
    width: int = 512
    steps: Optional[int] = None
    guidance: Optional[float] = None
    images: int = 1
    seed: Optional[int] = None
    half: bool = False
    cpu_offload: bool = False
    seq_offload: bool = False
    no_slicing: bool = False
    info_only: bool = False
    outdir: str = "outputs"
    # Video
    video: bool = False
    video_name: Optional[str] = None  # falls gesetzt überschreibt Standardnamen (run_id.mp4)
    video_frames: int = 0
    video_fps: int = 0
    video_blend_mode: str = "none"
    video_blend_steps: int = 0
    video_target_duration: float = 0.0  # gewünschte Videolänge in Sekunden (falls >0 überschreibt Heuristik)
    # Latent interpolation
    interp_seed_start: Optional[int] = None
    interp_seed_end: Optional[int] = None
    interp_frames: int = 0
    interp_slerp: bool = False
    # Seed cycling
    seed_cycle: int = 0
    seed_step: int = 997
    latent_jitter: float = 0.0
    # Morphing
    morph_from: Optional[str] = None
    morph_to: Optional[str] = None
    morph_frames: int = 0
    morph_seed_start: Optional[int] = None
    morph_seed_end: Optional[int] = None
    morph_latent: bool = False
    morph_slerp: bool = False
    # Run directory management
    make_run_dir: bool = True
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    write_meta: bool = True
    # Internal / dynamic
    mode: str = "single"  # wird zur Laufzeit gesetzt: single|batch|seed_cycle|interpolation|morph

