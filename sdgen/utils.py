import os, time, re, json, platform
from dataclasses import asdict

def timestamp() -> str:
    return time.strftime('%Y%m%d-%H%M%S')

def sanitize_prompt(p: str, max_len: int = 40) -> str:
    p = p.strip().lower()
    repl = {"ä":"ae","ö":"oe","ü":"ue","ß":"ss"}
    for k,v in repl.items():
        p = p.replace(k,v)
    p = re.sub(r'[^a-z0-9]+','-', p)
    p = re.sub(r'-+','-', p).strip('-') or 'prompt'
    if len(p) > max_len:
        p = p[:max_len].rstrip('-')
    return p

def build_run_id(prompt: str) -> str:
    return f"{timestamp()}-{sanitize_prompt(prompt)}"

def build_morph_run_id(from_prompt: str, to_prompt: str) -> str:
    return f"{timestamp()}-{sanitize_prompt(from_prompt)}-to-{sanitize_prompt(to_prompt)}"

def write_metadata(run_dir: str, run_id: str, meta: dict):
    path = os.path.join(run_dir, f"{run_id}.json")
    try:
        with open(path,'w',encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None

def find_ffmpeg():
    """Find FFmpeg executable - prioritize local project installation"""
    import shutil
    
    # Get project root directory (go up from sdgen to project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Priority order: Local project FFmpeg first, then system installations
    possible_paths = [
        # Local project FFmpeg (highest priority)
        os.path.join(project_root, "ffmpeg", "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"),
        os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(project_root, "bin", "ffmpeg.exe"),
        os.path.join(project_root, "ffmpeg.exe"),
        
        # System PATH
        "ffmpeg",  # Unix/Linux in PATH
        "ffmpeg.exe",  # Windows in PATH
        
        # Common Windows system installations
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe", 
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    
    for path in possible_paths:
        # Check if file exists directly (for absolute paths)
        if os.path.isfile(path):
            return path
        # Check if command exists in PATH (for relative commands)
        elif shutil.which(path):
            return shutil.which(path)
            
    return None

def save_image_with_meta(img, path: str, meta: dict):
    """Speichert PNG mit einfachen Text-Metadaten. Fällt still auf normales Speichern zurück."""
    try:
        from PIL import PngImagePlugin  # type: ignore
        pnginfo = PngImagePlugin.PngInfo()
        for k,v in meta.items():
            if v is None:
                continue
            try:
                pnginfo.add_text(str(k), str(v))
            except Exception:
                pass
        img.save(path, pnginfo=pnginfo)
    except Exception:
        img.save(path)
    return path
