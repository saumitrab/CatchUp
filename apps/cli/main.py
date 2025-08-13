from __future__ import annotations

import glob
import json
import os
import time
from typing import Optional

import numpy as np
import typer
from rich import print as rprint
from rich.prompt import Confirm
from rich.table import Table

from core.asr.whisper_rt import WhisperRealtime
from core.audio.loopback import AudioStream, find_device
from core.chunking.segment_merger import Seg, merge_segments
from core.embed.encoder import STEncoder
from core.llm.lmstudio_client import LMStudioClient
from core.rag.vector_store import ChromaStore
from core.storage.session import (
    SessionPaths,
    atomic_append_jsonl,
    create_session,
    update_meta_last_write,
)
from core.summarize.pipeline import map_reduce, simple
from core.utils.config import load_config
from core.utils.logging import setup_logging
from core.utils.time import now_iso
from core.audio.resample import resample_to_16k  # NEW: 16 kHz resampler

app = typer.Typer(add_completion=False)
logger = setup_logging()

DATA_DIR = "data/sessions"


def disclaimer_once():
    flag = os.path.expanduser("~/.catchup_disclaimer_accepted")
    if os.path.exists(flag):
        return True
    rprint(
        "[bold]CatchUp runs locally[/bold]. It transcribes audio routed from Chrome and stores only text on your disk.\n"
        "[yellow]Do not record content where you don't have rights.[/yellow]\n"
    )
    if Confirm.ask("Proceed and remember this choice?", default=True):
        with open(flag, "w") as f:
            f.write(now_iso())
        return True
    raise typer.Abort()


@app.command()
def start(
    session: str = typer.Option(..., help="Human-friendly session name"),
    model: str = typer.Option("small", help="ASR model: small|medium"),
    # ASR backend selection — leave None to auto-detect (Metal on macOS; CPU elsewhere)
    device: Optional[str] = typer.Option(
        None, help="ASR device override: metal|cpu|cuda|auto (default: auto-detect)"
    ),
    compute_type: Optional[str] = typer.Option(
        None, help="ASR compute type override (e.g., float16 on Metal, int8 on CPU)"
    ),
    # Audio input device match
    input_device: str = typer.Option(
        "BlackHole 2ch", help='Input device name contains this string (e.g., "BlackHole 2ch")'
    ),
    autosave_sec: int = typer.Option(10, help="Autosave frequency in seconds (metadata)"),
    vad: str = typer.Option("off", help="(placeholder for future VAD) on|off"),
    llm_model: str = typer.Option("llama-3.1-8b-instruct", help="LM Studio model name"),
    embed_model: str = typer.Option("all-MiniLM-L6-v2", help="Sentence-Transformers model"),
    lang: str = typer.Option("en", help="Language code, or 'auto' to autodetect"),
):
    """
    Start capturing audio from the loopback device, transcribe in rolling chunks,
    and append segments to transcript.jsonl. No audio is written to disk.
    """
    disclaimer_once()
    cfg = load_config()
    sp: SessionPaths = create_session(session)
    rprint(f"[green]Session created:[/green] {sp.session_id}")

    # Update meta with chosen models
    with open(sp.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["asr"]["model"] = model
    meta["embedding"]["model"] = embed_model
    meta["llm"]["model"] = llm_model
    with open(sp.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Resolve input (loopback) device
    device_index = find_device(input_device)
    if device_index is None:
        rprint(
            "[red]Loopback input device not found.[/red] "
            "Install BlackHole 2ch and route Chrome output to it:\n"
            "  brew install blackhole-2ch"
        )
        raise typer.Exit(code=20)

    samplerate = int(cfg.get("audio", {}).get("sample_rate", 48000))
    channels = int(cfg.get("audio", {}).get("channels", 2))
    chunk_seconds = int(cfg.get("asr", {}).get("chunk_seconds", 30))
    chunk_overlap = int(cfg.get("asr", {}).get("chunk_overlap_seconds", 3))
    beam = int(cfg.get("asr", {}).get("beam_size", 1))

    # Initialize ASR with robust auto device/compute selection
    asr = WhisperRealtime(
        model_name=model,
        device=device,            # None -> auto
        compute_type=compute_type,  # None -> auto
        language=(None if lang == "auto" else lang),
        beam_size=beam,
    )

    frames_per_chunk = samplerate * chunk_seconds
    frames_overlap = samplerate * chunk_overlap
    buf: list[np.ndarray] = []
    offset_sec = 0.0
    seg_count = 0

    rprint("[bold green]Recording...[/bold green] Use `catchup stop` (or the module command) from another terminal to end.")

    try:
        last_meta_save = time.time()
        with AudioStream(device_index, samplerate, channels) as stream:
            while True:
                # cooperative stop
                if os.path.exists(sp.stop_flag):
                    rprint("[yellow]Stop flag detected. Finalizing...[/yellow]")
                    break

                # Read ~1 second of audio into memory (float32, mono-left)
                frames = stream.read(int(samplerate * 1))
                buf.append(frames.copy())

                total = sum(len(x) for x in buf)
                if total >= frames_per_chunk:
                    # --- build a single contiguous chunk at capture rate (e.g., 48k) ---
                    concat = np.concatenate(buf, axis=0).astype("float32")

                    # RESAMPLE to 16 kHz for Whisper arrays
                    audio16 = resample_to_16k(concat, samplerate)

                    # Transcribe this chunk with the current stream offset (offset is wall time in seconds)
                    for seg in asr.transcribe_chunk(audio16, offset_sec):
                        seg_count += 1
                        rec = {
                            "seg_id": seg.seg_id,
                            "t_start": round(seg.t_start, 2),
                            "t_end": round(seg.t_end, 2),
                            "text": seg.text,
                            "avg_logprob": seg.avg_logprob,
                        }
                        atomic_append_jsonl(sp.transcript_jsonl, rec)

                    # Slide the buffer: keep only overlap tail (in capture-rate samples)
                    keep = frames_overlap
                    processed = max(0, len(concat) - keep)
                    offset_sec += processed / samplerate
                    if keep > 0 and len(concat) > keep:
                        buf = [concat[-keep:]]
                    else:
                        buf = []

                # periodic meta heartbeat
                now = time.time()
                if now - last_meta_save >= max(1, autosave_sec):
                    update_meta_last_write(sp.meta)
                    last_meta_save = now

                time.sleep(0.01)

    except KeyboardInterrupt:
        rprint("[yellow]Interrupted by user. Finalizing...[/yellow]")

    # finalize
    rprint(f"[green]Recording finished.[/green] Segments written: ~{seg_count}")
    if os.path.exists(sp.stop_flag):
        os.remove(sp.stop_flag)


@app.command()
def stop(session_id: Optional[str] = typer.Option(None, help="Session id (if omitted, last session)")):
    """Signal the running recorder to stop gracefully."""
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
    """Print meta.json for the session."""
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


@app.command("list")
def list_sessions():
    """List known sessions."""
    sessions = sorted(glob.glob(os.path.join(DATA_DIR, "*")))
    table = Table(title="CatchUp Sessions")
    table.add_column("Session ID")
    table.add_column("Created At")
    for s in sessions:
        meta = os.path.join(s, "meta.json")
        created = "n/a"
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                created = json.load(f).get("created_at", "n/a")
        table.add_row(os.path.basename(s), created)
    rprint(table)


def _load_segments(path: str):
    if not os.path.exists(path):
        return []
    segs: list[Seg] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            segs.append(
                Seg(
                    seg_id=obj["seg_id"],
                    t_start=obj["t_start"],
                    t_end=obj["t_end"],
                    text=obj["text"],
                )
            )
    return segs


@app.command()
def summarize(
    session_id: str = typer.Option(...),
    strategy: str = typer.Option("map-reduce", help="map-reduce|simple"),
    out: Optional[str] = None,
):
    """Summarize the session transcript via local LLM (LM Studio)."""
    base = os.path.join(DATA_DIR, session_id)
    tpath = os.path.join(base, "transcript.jsonl")
    segs = _load_segments(tpath)
    if not segs:
        rprint("[red]No transcript found.[/red]")
        raise typer.Exit(code=2)

    chunks = merge_segments(segs)
    llm = LMStudioClient()

    if strategy == "simple":
        text = " ".join(s.text for s in segs)
        summary = simple(text, llm)
    else:
        texts = [c.text for c in chunks]
        summary = map_reduce(texts, llm)

    out_dir = os.path.join(base, "exports")
    os.makedirs(out_dir, exist_ok=True)
    out = out or os.path.join(out_dir, "summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary)
    rprint(f"[green]Summary written:[/green] {out}")


@app.command()
def ask(
    session_id: str = typer.Option(...),
    question: str = typer.Argument(...),
    k: int = 6,
    out: Optional[str] = None,
):
    """Answer a question strictly from the session transcript via local RAG + LLM."""
    base = os.path.join(DATA_DIR, session_id)
    tpath = os.path.join(base, "transcript.jsonl")
    segs = _load_segments(tpath)
    if not segs:
        rprint("[red]No transcript found.[/red]")
        raise typer.Exit(code=2)

    # Build chunks + embeddings and upsert into Chroma (per-session)
    chunks = merge_segments(segs)
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
    answer = llm.chat([{"role": "user", "content": prompt}])

    out_dir = os.path.join(base, "exports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out or os.path.join(out_dir, "answer.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)
    rprint(f"[green]Answer written:[/green] {out_path}")


@app.command()
def export(
    session_id: str = typer.Option(...),
    formats: str = typer.Option("txt", help="Comma-separated: txt,json,srt,vtt"),
    out_dir: Optional[str] = None,
):
    """Export transcript to TXT/JSON/SRT/VTT."""
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

    def fmt_srt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    if "srt" in fmts:
        with open(os.path.join(out_dir, "transcript.srt"), "w", encoding="utf-8") as f:
            for i, s in enumerate(segs, 1):
                f.write(
                    f"{i}\n{fmt_srt_time(s.t_start)} --> {fmt_srt_time(s.t_end)}\n{s.text}\n\n"
                )

    if "vtt" in fmts:
        with open(os.path.join(out_dir, "transcript.vtt"), "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for s in segs:
                start = fmt_srt_time(s.t_start).replace(",", ".")
                end = fmt_srt_time(s.t_end).replace(",", ".")
                f.write(f"{start} --> {end}\n{s.text}\n\n")

    rprint(f"[green]Exports written to:[/green] {out_dir}")


@app.command()
def clean(session_id: str = typer.Option(...), force: bool = False):
    """Remove derived data (vector store, exports). Keep transcript unless --force."""
    base = os.path.join(DATA_DIR, session_id)
    removed = False
    import shutil

    chroma = os.path.join(base, "chroma")
    if os.path.isdir(chroma):
        shutil.rmtree(chroma)
        removed = True

    exports = os.path.join(base, "exports")
    if os.path.isdir(exports) and force:
        shutil.rmtree(exports)
        removed = True

    rprint("[green]Cleaned derived files.[/green]" if removed else "[yellow]Nothing to clean.[/yellow]")


@app.command()
def devices():
    """List input-capable audio devices (for --input-device matching)."""
    import sounddevice as sd

    devs = sd.query_devices()
    table = Table(title="Input devices")
    table.add_column("Index")
    table.add_column("Name")
    table.add_column("Max In")
    table.add_column("Default SR")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            table.add_row(
                str(i),
                d.get("name", ""),
                str(d.get("max_input_channels", 0)),
                str(d.get("default_samplerate", "")),
            )
    rprint(table)


if __name__ == "__main__":
    app()
