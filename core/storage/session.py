from __future__ import annotations
import os, json, tempfile, shutil
from dataclasses import dataclass
from typing import Dict, Any
from ..utils.time import now_iso, session_id as make_session_id

BASE_DIR = "data/sessions"

@dataclass
class SessionPaths:
    session_id: str
    base: str
    meta: str
    transcript_jsonl: str
    transcript_index: str
    chunks_jsonl: str
    chroma_dir: str
    exports_dir: str
    visuals_dir: str
    stop_flag: str
    log: str

def create_session(name: str) -> SessionPaths:
    sid = make_session_id(name)
    base = os.path.join(BASE_DIR, sid)
    os.makedirs(base, exist_ok=True)
    sp = SessionPaths(
        session_id=sid,
        base=base,
        meta=os.path.join(base, "meta.json"),
        transcript_jsonl=os.path.join(base, "transcript.jsonl"),
        transcript_index=os.path.join(base, "transcript.index.json"),
        chunks_jsonl=os.path.join(base, "chunks.jsonl"),
        chroma_dir=os.path.join(base, "chroma"),
        exports_dir=os.path.join(base, "exports"),
        visuals_dir=os.path.join(base, "visuals"),
        stop_flag=os.path.join(base, "stop.flag"),
        log=os.path.join(base, "session.log"),
    )
    os.makedirs(sp.exports_dir, exist_ok=True)
    os.makedirs(sp.visuals_dir, exist_ok=True)
    # meta
    meta = {
        "session_id": sid,
        "name": name,
        "created_at": now_iso(),
        "source_app": "chrome",
        "lang": "en",
        "asr": {"engine": "faster-whisper", "model": "small", "word_timestamps": False},
        "embedding": {"engine": "sentence-transformers", "model": "all-MiniLM-L6-v2", "dim": 384},
        "llm": {"provider": "lmstudio", "base_url": "http://localhost:1234/v1", "model": "llama-3.1-8b-instruct"},
        "storage": {"keep_audio": False, "format": "jsonl"},
        "status": {"recording": True, "last_write": None},
    }
    atomic_write(sp.meta, meta)
    return sp

def atomic_write(path: str, obj: Any):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def atomic_append_jsonl(path: str, obj: Any):
    # atomic append by writing a temp and concatenating
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def update_meta_last_write(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["status"]["last_write"] = now_iso()
    atomic_write(meta_path, meta)
