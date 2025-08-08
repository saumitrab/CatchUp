from __future__ import annotations
import typer, os, json, time, threading, queue, logging, glob
from typing import Optional
from rich import print as rprint
from rich.prompt import Confirm
from rich.table import Table

from core.utils.logging import setup_logging
from core.utils.config import load_config
from core.utils.time import now_iso
from core.storage.session import create_session, atomic_append_jsonl, update_meta_last_write, SessionPaths
from core.audio.loopback import find_device, AudioStream
from core.asr.whisper_rt import WhisperRealtime, Segment
from core.chunking.segment_merger import Seg, merge_segments
from core.embed.encoder import STEncoder
from core.rag.vector_store import ChromaStore
from core.llm.lmstudio_client import LMStudioClient
from core.summarize.pipeline import map_reduce, simple

app = typer.Typer(add_completion=False)
logger = setup_logging()

DATA_DIR = "data/sessions"

def disclaimer_once():
    flag = os.path.expanduser("~/.catchup_disclaimer_accepted")
    if os.path.exists(flag):
        return True
    rprint("[bold]CatchUp runs locally[/bold]. It transcribes audio routed from Chrome and stores only text on your disk.\n"
           "[yellow]Do not record content where you don't have rights.[/yellow]\n")
    if Confirm.ask("Proceed and remember this choice?", default=True):
        with open(flag, "w") as f:
            f.write(now_iso())
        return True
    raise typer.Abort()

@app.command()
def start(session: str = typer.Option(..., help="Human-friendly session name"),
          model: str = typer.Option("small", help="ASR model: small|medium"),
          device: str = typer.Option("BlackHole", help="Device name contains..."),
          autosave_sec: int = typer.Option(10, help="Autosave frequency in seconds"),
          vad: str = typer.Option("off", help="(placeholder) VAD on|off"),
          llm_model: str = typer.Option("llama-3.1-8b-instruct", help="LM Studio model name"),
          embed_model: str = typer.Option("all-MiniLM-L6-v2", help="Sentence-Transformers model"),
         ):
    disclaimer_once()
    cfg = load_config()
    sp: SessionPaths = create_session(session)
    rprint(f"[green]Session created:[/green] {sp.session_id}")
    # write meta update for ASR & LLM choices
    with open(sp.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["asr"]["model"] = model
    meta["embedding"]["model"] = embed_model
    meta["llm"]["model"] = llm_model
    with open(sp.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # device selection
    device_index = find_device(device)
    if device_index is None:
        rprint("[red]BlackHole input device not found.[/red] Install with: brew install blackhole-2ch")
        raise typer.Exit(code=20)

    samplerate = int(cfg.get("audio",{}).get("sample_rate", 48000))
    channels = int(cfg.get("audio",{}).get("channels", 2))
    chunk_seconds = int(cfg.get("asr",{}).get("chunk_seconds", 30))
    chunk_overlap = int(cfg.get("asr",{}).get("chunk_overlap_seconds", 3))

    asr = WhisperRealtime(model_name=model, compute_type="int8_float16", language="en", beam_size=int(cfg.get("asr",{}).get("beam_size",1)))

    # recording loop: read frames, build rolling chunks, transcribe each chunk
    frames_per_chunk = samplerate * chunk_seconds
    frames_overlap = samplerate * chunk_overlap
    buf = []
    offset_sec = 0.0
    seg_count = 0

    rprint("[bold green]Recording...[/bold green] Press `catchup stop` in another terminal to end.")

    try:
        with AudioStream(device_index, samplerate, channels) as stream:
            while True:
                if os.path.exists(sp.stop_flag):
                    rprint("[yellow]Stop flag detected. Finalizing...[/yellow]")
                    break
                frames = stream.read(int(samplerate * 1))  # 1 second chunks
                buf.append(frames.copy())
                total = sum(len(x) for x in buf)
                if total >= frames_per_chunk:
                    # build chunk array
                    arr = buf.copy()
                    audio_np = arr[0]
                    for a in arr[1:]:
                        # concatenate
                        audio_np = audio_np if audio_np.size == 0 else audio_np
                        audio_np = (audio_np, a) if isinstance(audio_np, tuple) else (audio_np, a)
                    # Properly concatenate arrays
                    import numpy as np
                    audio_np = np.concatenate(buf, axis=0).astype("float32")

                    # Transcribe
                    for seg in asr.transcribe_chunk(audio_np, offset_sec):
                        seg_count += 1
                        rec = {
                            "seg_id": seg.seg_id,
                            "t_start": round(seg.t_start, 2),
                            "t_end": round(seg.t_end, 2),
                            "text": seg.text,
                            "avg_logprob": seg.avg_logprob,
                        }
                        atomic_append_jsonl(sp.transcript_jsonl, rec)
                        update_meta_last_write(sp.meta)

                    # slide buffer with overlap
                    # keep last 'frames_overlap' frames
                    keep = frames_overlap
                    concat = np.concatenate(buf, axis=0)
                    if keep > 0 and len(concat) > keep:
                        buf = [concat[-keep:]]
                    else:
                        buf = []
                    offset_sec += (len(concat) - (len(buf[0]) if buf else 0)) / samplerate
                time.sleep(0.05)
    except KeyboardInterrupt:
        rprint("[yellow]Interrupted by user. Finalizing...[/yellow]")

    # finalize
    rprint(f"[green]Recording finished.[/green] Segments written: ~{seg_count}")
    # remove stop flag
    if os.path.exists(sp.stop_flag):
        os.remove(sp.stop_flag)

@app.command()
def stop(session_id: Optional[str] = typer.Option(None, help="Session id (if omitted, last session)")):
    # create stop.flag in target session
    target = None
    if session_id:
        target = os.path.join(DATA_DIR, session_id)
    else:
        sessions = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
        target = sessions[-1] if sessions else None
    if not target or not os.path.isdir(target):
        rprint("[red]No session found.[/red]")
        raise typer.Exit(code=1)
    flag = os.path.join(target, "stop.flag")
    with open(flag, "w") as f:
        f.write(now_iso())
    rprint(f"[green]Stop signal written:[/green] {flag}")

@app.command()
def status(session_id: Optional[str] = typer.Option(None, help="Session id (if omitted, last session)")):
    target = None
    if session_id:
        target = os.path.join(DATA_DIR, session_id)
    else:
        sessions = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
        target = sessions[-1] if sessions else None
    if not target or not os.path.isdir(target):
        rprint("[red]No session found.[/red]")
        raise typer.Exit(code=1)
    meta = os.path.join(target, "meta.json")
    if not os.path.exists(meta):
        rprint("[red]Missing meta.json[/red]")
        raise typer.Exit(code=2)
    with open(meta, "r", encoding="utf-8") as f:
        m = json.load(f)
    rprint(m)

@app.command()
def list():
    sessions = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
    table = Table(title="CatchUp Sessions")
    table.add_column("Session ID")
    table.add_column("Created At")
    for s in sessions:
        meta = os.path.join(s, "meta.json")
        created = "n/a"
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                created = json.load(f).get("created_at","n/a")
        table.add_row(os.path.basename(s), created)
    rprint(table)

def _load_segments(path: str):
    if not os.path.exists(path):
        return []
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            segs.append(Seg(seg_id=obj["seg_id"], t_start=obj["t_start"], t_end=obj["t_end"], text=obj["text"]))
    return segs

@app.command()
def summarize(session_id: str = typer.Option(...), strategy: str = typer.Option("map-reduce"), out: Optional[str] = None):
    base = os.path.join(DATA_DIR, session_id)
    tpath = os.path.join(base, "transcript.jsonl")
    segs = _load_segments(tpath)
    if not segs:
        rprint("[red]No transcript found.[/red]")
        raise typer.Exit(code=2)
    # chunks
    chunks = merge_segments(segs)
    # LLM
    llm = LMStudioClient()
    if strategy == "simple":
        text = " ".join(s.text for s in segs)
        summary = simple(text, llm)
    else:
        texts = [c.text for c in chunks]
        summary = map_reduce(texts, llm)
    # write out
    out_dir = os.path.join(base, "exports")
    os.makedirs(out_dir, exist_ok=True)
    out = out or os.path.join(out_dir, "summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary)
    rprint(f"[green]Summary written:[/green] {out}")

@app.command()
def ask(session_id: str = typer.Option(...), question: str = typer.Argument(...), k: int = 6, out: Optional[str] = None):
    base = os.path.join(DATA_DIR, session_id)
    tpath = os.path.join(base, "transcript.jsonl")
    segs = _load_segments(tpath)
    if not segs:
        rprint("[red]No transcript found.[/red]")
        raise typer.Exit(code=2)
    # ensure Chroma is built
    chunks = merge_segments(segs)
    from core.embed.encoder import STEncoder
    from core.rag.vector_store import ChromaStore
    enc = STEncoder()
    embs = enc.embed_texts([c.text for c in chunks])
    store = ChromaStore(persist_directory=os.path.join(base, "chroma"))
    store.upsert(chunks, embs)
    hits = store.query(question, k=k)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    prompt = (
        "Answer strictly from the provided transcript excerpts. If unsure, say you're not sure.\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    llm = LMStudioClient()
    answer = llm.chat([{"role":"user","content":prompt}])
    out_dir = os.path.join(base, "exports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out or os.path.join(out_dir, "answer.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)
    rprint(f"[green]Answer written:[/green] {out_path}")

@app.command()
def export(session_id: str = typer.Option(...), formats: str = typer.Option("txt"), out_dir: Optional[str] = None):
    base = os.path.join(DATA_DIR, session_id)
    tpath = os.path.join(base, "transcript.jsonl")
    segs = _load_segments(tpath)
    if not segs:
        rprint("[red]No transcript found.[/red]")
        raise typer.Exit(code=2)
    out_dir = out_dir or os.path.join(base, "exports")
    os.makedirs(out_dir, exist_ok=True)
    fmts = [f.strip().lower() for f in formats.split(",")]
    if "txt" in fmts:
        with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            for s in segs:
                f.write(s.text + "\n")
    if "json" in fmts:
        with open(os.path.join(out_dir, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump([s.__dict__ for s in segs], f, indent=2)
    if "srt" in fmts or "vtt" in fmts:
        def fmt_time(t):
            import math
            h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t-int(t))*1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        if "srt" in fmts:
            with open(os.path.join(out_dir, "transcript.srt"), "w", encoding="utf-8") as f:
                for i, s in enumerate(segs, 1):
                    f.write(f"{i}\n{fmt_time(s.t_start)} --> {fmt_time(s.t_end)}\n{s.text}\n\n")
        if "vtt" in fmts:
            with open(os.path.join(out_dir, "transcript.vtt"), "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for i, s in enumerate(segs, 1):
                    f.write(f"{fmt_time(s.t_start).replace(',', '.')} --> {fmt_time(s.t_end).replace(',', '.')}\n{s.text}\n\n")
    rprint(f"[green]Exports written to:[/green] {out_dir}")

@app.command()
def clean(session_id: str = typer.Option(...), force: bool = False):
    base = os.path.join(DATA_DIR, session_id)
    chroma = os.path.join(base, "chroma")
    removed = False
    import shutil
    if os.path.isdir(chroma):
        shutil.rmtree(chroma)
        removed = True
    exports = os.path.join(base, "exports")
    if os.path.isdir(exports) and force:
        shutil.rmtree(exports)
        removed = True
    rprint("[green]Cleaned derived files.[/green]" if removed else "[yellow]Nothing to clean.[/yellow]")
