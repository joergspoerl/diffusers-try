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

def write_metadata(run_dir: str, run_id: str, meta: dict):
    path = os.path.join(run_dir, f"{run_id}.json")
    try:
        with open(path,'w',encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None
