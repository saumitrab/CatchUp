# CatchUp — PRD & Implementation Plan (v0 → v3)

> **Goal:** Build a local macOS utility that captures **Chrome**’s playing audio, transcribes to text with timestamps (no audio stored), indexes it for retrieval, and enables local LLM summaries & Q&A — starting as a CLI, then optional UI and visuals. Written so an LLM can implement a working repo directly.

---

## 0) Project metadata

- **Name:** CatchUp
- **CLI command:** `catchup`
- **License:** MIT (open source, public repo is OK)
- **Target machine:** Mac mini M4, 16GB RAM, 256GB storage
- **Platforms:** macOS (Apple Silicon/Metal)
- **Privacy:** 100% local by default (no network). Optional cloud toggles arrive in **v1** but are off unless configured.
- **Source of audio:** **Google Chrome** only (MVP assumption).
- **No audio persistence:** Never store raw audio; process in-memory and discard once transcribed.
- **Language:** English only (MVP).

---

## 1) Roadmap

### v0 — CLI, fully local (MVP)
- Capture system output via loopback device (BlackHole 2ch) and **live-transcribe** using **faster-whisper** (CTranslate2 + Metal/MPS).
- Persist **JSONL transcript** (segment-level timestamps). Optional TXT/SRT/VTT export on demand.
- Chunk & embed transcript locally (**sentence-transformers**; **Chroma** vector store).
- Local LLM via **LM Studio** (OpenAI-compatible API) for:
  - **Conference-buddy** style summaries (map–reduce over chunks).
  - **Local RAG Q&A** strictly over transcript (no web search).
- Crash-safe rolling writes, autosave, and **resume** by session id.
- Simple CLI: `start`, `stop`, `summarize`, `ask`, `export`, `status`, `list`, `clean`.

### v1 — Optional cloud toggles (still CLI)
- Add configuration to optionally route LLM/embeddings to remote providers **if** API keys are provided (OpenAI, etc.).
- Defaults remain **local-only**. **No web search** yet (explicitly deferred).

### v2 — Streamlit UI
- Basic UI wrapper: Start/Stop, live transcript tail, “Summarize”, “Ask”, and Export.
- Uses the same underlying pipeline as CLI.

### v3 — Visuals (screenshots)
- Request **Screen Recording** permission and capture **screenshots** when the transcript suggests (“as you can see…”, “on this slide…”). Save **WebP**; link images to nearest transcript timestamp. No OCR yet (v3.1).

---

## 2) User stories & success criteria

### Primary user
Someone who can’t attend a long stream (YouTube, conference) and wants a **local** capture → summary → ask follow-ups later.

### Stories
1. **Start capture (Chrome-only):** `catchup start --session "APAC Summit Keynote"` → tool prints a one-time disclaimer about recording rights and starts live transcription.
2. **Stop & summarize:** `catchup stop` → `catchup summarize --session <id>` produces a concise, structured summary.
3. **Ask clarifying questions:** `catchup ask --session <id> "What did they say about multi-tenant safety?"` → answer based solely on transcript context.

### Success criteria
- 8-hour run completes with **no crash** on the target hardware.
- Live transcription lags real-time by < ~8 seconds using **small** or **medium** model.
- Summaries are clear, actionable, and readable without needing the raw transcript.

---

## 3) Constraints & assumptions

- **Chrome-only** supported for MVP. Operationally, route Chrome output to **BlackHole 2ch** (user action). Other apps should be muted/ignored.
- **No audio files** are written to disk at any point.
- **Segment-level timestamps** (not word-level) are sufficient.
- **Rolling processing** with periodic autosave.
- **No tests/CI** in MVP (can be added later). No packaging/Homebrew formula initially; source + README install only.
- All data remains **local**. No encryption/redaction requirements initially.
- Each recording is stored under a named **session** (prompted at start).

---

## 4) Architecture (v0/v1 core)

```
[Chrome audio output to BlackHole 2ch]
       │
       ▼
[sounddevice PortAudio stream]  →  [ASR worker - faster-whisper (Metal)]
       │                                      │
       │                                      ▼
       │                              [Segment JSONL write]
       │                                      │
       └──────────────→ [Chunker → Embeddings (Sentence-Transformers)] → [Chroma]
                                                     │
                          ┌───────────────────────────┴───────────────────────────┐
                          ▼                                                       ▼
                 [Summarizer via LM Studio]                                [RAG Q&A via LM Studio]
```

**Notes**
- **ASR:** `faster-whisper` (CT2). Default model: `small`. Allow `medium`. `language="en"`, `word_timestamps=False`.
- **Embeddings:** `sentence-transformers` (e.g., `all-MiniLM-L6-v2`), MPS if available.
- **Vector store:** **Chroma** (session-local persistence).
- **LLM:** **LM Studio** exposing OpenAI-compatible API (e.g., `http://localhost:1234/v1`). Model name set in config (e.g., `llama-3.1-8b-instruct`).

---

## 5) Setup & dependencies

### macOS prerequisites (manual)
1. Install **Homebrew** (if not present).
2. Install **BlackHole 2ch** loopback device:
   - `brew install blackhole-2ch`
   - In **System Settings → Sound**, set **Chrome** output to **BlackHole 2ch** (or create a Multi-Output so you can still hear audio).
3. Python 3.12 recommended; create a virtualenv.

### Python libs (v0)
- `faster-whisper` (CTranslate2 backend with Metal)
- `numpy`, `sounddevice`
- `typer`, `rich`, `pydantic`
- `sentence-transformers`, `chromadb`
- `requests` (LM Studio client)

> Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to improve Apple Silicon compatibility.

---

## 6) Data & file layout

```
repo/
  apps/
    cli/                 # Typer entrypoint
    ui/                  # (v2 Streamlit)
    visuals/             # (v3 screenshots)
  core/
    audio/               # capture + device mgmt
    asr/                 # faster-whisper runner
    chunking/            # text chunker + sectionizer
    embed/               # embeddings + Chroma adapters
    rag/                 # retrieval + ranking
    llm/                 # LM Studio/OpenAI-compatible clients
    summarize/           # map-reduce pipeline + prompts
    storage/             # session store, atomic writers
    utils/               # logging, config, time
  configs/
    default.yaml
  data/
    sessions/
      <session-id>/
        meta.json
        transcript.jsonl
        transcript.index.json
        chunks.jsonl
        chroma/
        exports/
        visuals/            # (v3) webp images
  docs/
    PRD.md
  README.md
  pyproject.toml
  .python-version
```

### Session naming
- Prompt the user for a **session name** at start (e.g., `"APAC Summit Keynote"`).
- Internal **session id**: `YYYYMMDD-HHMM-<kebab(name)>-<shortid>`.

---

## 7) Storage formats

### 7.1 `meta.json`
```json
{
  "session_id": "20250807-0930-apac-summit-keynote-y12f",
  "name": "APAC Summit Keynote",
  "created_at": "2025-08-07T09:30:00-07:00",
  "source_app": "chrome",
  "lang": "en",
  "asr": { "engine": "faster-whisper", "model": "small", "word_timestamps": false },
  "embedding": { "engine": "sentence-transformers", "model": "all-MiniLM-L6-v2", "dim": 384 },
  "llm": { "provider": "lmstudio", "base_url": "http://localhost:1234/v1", "model": "llama-3.1-8b-instruct" },
  "storage": { "keep_audio": false, "format": "jsonl" },
  "status": { "recording": true, "last_write": "2025-08-07T10:15:23-07:00" }
}
```

### 7.2 `transcript.jsonl` (one JSON object per line)
```json
{"seg_id":"seg-000001","t_start":12.43,"t_end":18.72,"text":"Welcome everyone to the APAC Summit...","avg_logprob":-0.12}
```

### 7.3 `chunks.jsonl`
```json
{"chunk_id":"chk-000034","seg_ids":["seg-000120","seg-000121","seg-000122"],"t_start":3421.1,"t_end":3488.9,"text":"...merged text...","token_count":1420}
```

---

## 8) Chunking strategy

- **Live segmentation** follows ASR segments (silence boundaries).
- **Post segmentation** (for RAG/summaries):
  - Merge adjacent segments until **~1200–1600 tokens**, with **~10% overlap** between chunks (overlap by segment boundaries).
  - Start a new chunk on **topic drift**: cosine similarity of rolling embeddings drops below threshold (e.g., **0.80**).
  - Maintain `chunk_id → seg_id[]` mapping with `t_start/t_end` for internal traceability.
- Retrieval: top-**k**=6 by cosine similarity, then **MMR** to diversify evidence.

---

## 9) Summarization & Q&A prompts

### 9.1 “Conference buddy” summarization

**Map (per chunk):**  
> You are a diligent colleague who attended this talk. Write concise bullet points of the **key claims**, **evidence/examples**, **definitions**, and **action items** in this chunk. Be factual. Do not hallucinate.

**Reduce (aggregate across chunks):**  
> Combine these chunk summaries into a single cohesive document with sections:  
> **TL;DR**, **Key Points**, **Examples/Case Studies**, **Takeaways/Actions**, **Open Questions**, **Suggested Follow‑ups** (queries the user could search later). Use only the provided content; no external info.

### 9.2 Local RAG Q&A

> Answer **strictly** from the provided transcript excerpts. If unsure, say you’re not sure. Return a short answer. (Timestamps are retained internally but do not need to be surfaced unless requested.)

**LLM client** calls LM Studio’s OpenAI-compatible `/v1/chat/completions` with `model`, `messages`, `temperature`, `max_tokens`, `stop` (optional).

---

## 10) CLI specification (v0/v1)

**Typer**-based subcommands:

```
catchup start --session "<name>" [--model small|medium] [--device "BlackHole 2ch"]
              [--autosave-sec 10] [--vad on|off]
              [--llm-model "llama-3.1-8b-instruct"] [--embed-model "all-MiniLM-L6-v2"]

catchup stop [--session-id <id>]

catchup status [--session-id <id>]

catchup list

catchup summarize --session-id <id> [--strategy map-reduce|simple]
                  [--out data/sessions/<id>/exports/summary.txt]

catchup ask --session-id <id> "<question>"
            [--k 6] [--out data/sessions/<id>/exports/answer.txt]

catchup export --session-id <id> [--formats txt,json,srt,vtt]
               [--out-dir data/sessions/<id>/exports]

catchup clean --session-id <id> [--force]
```

**Defaults**
- `--model small`
- `--autosave-sec 10`
- Auto-detect device containing `BlackHole` (print setup instructions if missing).
- Summarize strategy defaults to **map-reduce**.
- Q&A top-k defaults to **6**.

**Behaviors**
- `start`: Creates session dir + `meta.json`, prints a one-time **disclaimer** (“record only where permitted; data stays local”), begins capture, transcribes live, appends JSONL segments atomically.
- `stop`: Signals workers, flushes buffers, finalizes chunking & embeddings.
- `summarize`: Builds/uses Chroma, runs map–reduce summary via LM Studio, writes `exports/summary.txt`.
- `ask`: Runs retrieval + LLM answer, writes optional file.
- `export`: Concatenates transcript to `.txt`; builds `.srt`/`.vtt`.
- `status`: Shows recording state, last segment time, segment count.
- `clean`: Removes Chroma and derived files but **keeps** transcript unless `--force`.

---

## 11) Modules & key functions (v0)

### `core/audio/loopback.py`
- `list_input_devices() -> list[Device]`
- `open_stream(device_name: str, sample_rate: int) -> Stream`
- Uses `sounddevice` to read from **BlackHole 2ch**; frames queued to ASR. No disk writes of audio.

### `core/asr/whisper_rt.py`
- `class WhisperRT:`
  - `__init__(model="small", device="auto", compute_type="int8_float16")`
  - `transcribe_stream(frames_iter) -> Iterator[Segment]`
- Wrap `faster_whisper.WhisperModel`; English-only; segment timestamps.

### `core/storage/session.py`
- `create_session(name: str) -> SessionPaths`
- `atomic_append_transcript(seg: Segment) -> None`
- `write_meta(...)`, `update_meta_status(...)`
- `build_srt_vtt(transcript)`

### `core/chunking/segment_merger.py`
- `merge_segments(segments, target_tokens=1400, overlap_ratio=0.1, sim_threshold=0.8) -> list[Chunk]`

### `core/embed/encoder.py`
- `class STEncoder:`
  - `__init__(model="all-MiniLM-L6-v2", device="mps"|"cpu")`
  - `embed_texts(texts: list[str]) -> np.ndarray`

### `core/rag/vector_store.py`
- `class ChromaStore:`
  - `upsert(chunks: list[Chunk])`
  - `query(query: str, k=6) -> list[ScoredChunk]`

### `core/llm/lmstudio_client.py`
- `chat(messages: list[dict], model: str, base_url="http://localhost:1234/v1", **opts) -> str`

### `core/summarize/pipeline.py`
- `map_reduce(chunks: list[Chunk], llm: LMClient) -> str`
- `simple(transcript: list[Segment], llm: LMClient) -> str`

### `apps/cli/main.py`
- Entry for all subcommands; Rich console output; clear errors & help.

---

## 12) Reliability & performance

- **Autosave** transcript every N seconds (default 10s) using atomic temp-file rename.
- **Workers:** `audio_capture` and `asr_runner` threads with a bounded queue.
- **Backpressure:** If ASR lags, drop oldest PCM frames conservatively (log once) rather than unbounded growth.
- **Resume:** If `start` is run on an existing session id, append from the last `seg_id`.
- **Notifications:** Console messages by default; optional macOS notifications behind a flag (v1).
- **Compute:** Default ASR model `small`; `medium` available. CTranslate2 `compute_type="int8_float16"` on Metal.

---

## 13) Config

### `configs/default.yaml`
```yaml
audio:
  device_preferred: "BlackHole 2ch"
  sample_rate: 48000
  channels: 2
  autosave_sec: 10

asr:
  model: "small"            # or "medium"
  language: "en"
  word_timestamps: false

embeddings:
  model: "all-MiniLM-L6-v2"
  dim: 384

llm:
  provider: "lmstudio"
  base_url: "http://localhost:1234/v1"
  model: "llama-3.1-8b-instruct"
  temperature: 0.2
  max_tokens: 1200

rag:
  top_k: 6
  overlap_ratio: 0.1
  target_tokens: 1400
  sim_threshold: 0.8

storage:
  keep_audio: false
  format: "jsonl"
```

**Env overrides (examples)**
- `CATCHUP_LLM_BASE_URL`, `CATCHUP_LLM_MODEL`, `CATCHUP_EMBED_MODEL`, etc.

---

## 14) First-run disclaimer (console)

> **CatchUp runs locally.** It transcribes audio routed from Chrome and stores only text on your disk. **Do not record content where you don’t have rights.** By proceeding, you confirm you will follow applicable laws and terms of service.

(Press `y` to continue; remembers acceptance in a local file.)

---

## 15) v1 cloud toggles

- `.env` support. If `OPENAI_API_KEY` is set, `llm.provider=openai` routes chat calls to OpenAI; if `EMBED_PROVIDER=openai`, embeddings route accordingly. Default remains **lmstudio** + local embeddings.
- No web search integration in v1 by design.

---

## 16) v2 Streamlit outline

- **Panels:** Record/Stop, live transcript tail (tail `transcript.jsonl`), Summary (render with sections), Ask (textbox → responses), Exports.
- Implementation can call the same backend modules or invoke CLI via subprocess for simplicity.

---

## 17) v3 Visuals (screenshots)

- Request **Screen Recording** permission (system prompt).
- **Triggers:** heuristic keyphrases (“as you can see”, “on this slide”, “look at this diagram”) + cooldown (e.g., 45s); optional periodic (every N minutes).
- **Capture method:** call macOS `screencapture` CLI for simplicity.
- **Storage:** Save **WebP** under `visuals/` and write an index JSON mapping `image_path → nearest seg_id/timestamp`.
- **UI linkages:** In v2 UI, show image thumbnails next to transcript excerpts (later work).

---

## 18) Error handling

- **No loopback device:** Print instructions to install **BlackHole 2ch**; exit code `20`.
- **LM Studio not reachable:** Warn and suggest starting the server/model; exit code `30`.
- **Chroma lock/corruption:** Rebuild vector store automatically.
- **Low disk:** Warn below 5GB free; stop gracefully if necessary (configurable threshold).

---

## 19) Logging

- `data/sessions/<id>/session.log` with timestamps, levels, and component tags like `[AUDIO] [ASR] [RAG]` using `logging` + **rich**.

---

## 20) Security & privacy

- Local-only default; **no audio files** ever stored.
- Normal delete is sufficient (secure wipe optional later).

---

## 21) Implementation steps (for an LLM)

### Phase A — Scaffold
1. Create repo structure (Section 6). Initialize `pyproject.toml` with dependencies. Add `README.md` with setup/usage.
2. Add `configs/default.yaml`; implement config loader with env overrides.

### Phase B — Audio capture & ASR
1. `core/audio/loopback.py`: enumerate devices; open input stream on device name containing “BlackHole”; yield PCM frames (float32, 48kHz, stereo→mono mix if needed).
2. `core/asr/whisper_rt.py`: wrap `faster-whisper`. Accept frames iterator; yield segments `{seg_id, t_start, t_end, text, avg_logprob}`. English-only; segment timestamps.
3. `core/storage/session.py`: create session dir; write `meta.json`; atomic append to `transcript.jsonl` and update `transcript.index.json` (byte offsets).

### Phase C — Chunking & embeddings
1. `core/chunking/segment_merger.py`: merge segments to ~1400 tokens, ~10% overlap, topic-drift reset using rolling embedding cosine similarity.
2. `core/embed/encoder.py`: load `sentence-transformers` model (MPS if available); `embed_texts` batched.
3. `core/rag/vector_store.py`: create per-session **Chroma** DB; `upsert(chunks)` and `query(query,k)`.

### Phase D — LLM & pipelines
1. `core/llm/lmstudio_client.py`: OpenAI-compatible `/v1/chat/completions`. Expose `chat(messages, model, base_url, **opts)`.
2. `core/summarize/pipeline.py`: implement **map–reduce** using prompts from Section 9. Add `simple` fallback.
3. `rag` Q&A: retrieve top-k, format excerpts, call LLM with “answer strictly from text” guardrails.

### Phase E — CLI
1. `apps/cli/main.py`: implement Typer commands (`start`, `stop`, `status`, `list`, `summarize`, `ask`, `export`, `clean`). Share a SessionManager across commands.
2. First‑run disclaimer; Rich status updates; friendly error messages.

### Phase F — Hardening
1. Autosave & atomic writes; bounded queues; conservative frame dropping under backpressure; retries with backoff.
2. Manual acceptance test on a 2–3 hour YouTube session.

### Phase G — v1 toggles
1. `.env` parsing; provider factories for LLM/embeddings.

### Phase H — v2/v3 stubs
1. Streamlit minimal app (Start/Stop/Summary/Ask). 
2. Visuals stub using `screencapture` and timestamp linkage.

---

## 22) Acceptance checklist (manual)

- [ ] `catchup start` prints disclaimer and begins live transcription (Chrome routed to BlackHole).
- [ ] `catchup status` shows ongoing progress (seg count, last timestamp).
- [ ] `catchup stop` finalizes gracefully; transcript JSONL is complete.
- [ ] `catchup summarize` creates `exports/summary.txt` with required sections.
- [ ] `catchup ask "question"` returns a concise answer grounded in transcript.
- [ ] No `.wav`/`.flac` in the session folder.
- [ ] Long run (4–8h) completes without crash on the target machine.

---

## 23) README outline

- What it is / isn’t (local-only, Chrome source, no audio stored).
- Install: Homebrew + BlackHole, Python venv, `pip install -e .`.
- LM Studio: how to start server and set `CATCHUP_LLM_MODEL` (examples).
- Quickstart:
  ```bash
  catchup start --session "APAC Summit"
  # later...
  catchup stop
  catchup summarize --session-id <id>
  catchup ask --session-id <id> "What were the main trade-offs?"
  catchup export --session-id <id> --formats srt
  ```
- Troubleshooting: device not found; LM Studio not running; slow ASR.

---

## 24) Deferred items

- Web search (future).
- OCR for screenshots (v3.1).
- Tests/CI and packaging/Homebrew formula.
- Multi-app capture & per-app filtering.

---

## 25) Notes & decisions encoded

- Default ASR model: **small**; allow **medium**. Easy to switch via CLI/config.
- Live transcription (not post‑processing) so we never store audio.
- Segment timestamps retained internally but **not emphasized** in summaries/answers.
- Hardware: 16GB RAM is sufficient for `small`/`medium` with Metal acceleration.
- A disk usage threshold parameter can be provided (default off) to stop if exceeded.
