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
