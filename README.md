# CatchUp

Local macOS CLI that **listens to Chrome’s audio**, **transcribes in real time** (no audio stored), and lets you **summarize** and **ask follow‑ups** with a local LLM (LM Studio).

> ✅ Latest: **left‑channel mono**, **resample to 16 kHz** (for Whisper), explicit **`--input-device "BlackHole 2ch"`**, optional **`--lang auto`**, and robust device/compute fallbacks.

---

## How it works

![Architecture](docs/architecture.svg)

For an exec‑friendly view, see the swimlane: `docs/architecture_swimlane.png`.

**Pipeline (v0 / CLI):**
1. **Chrome → BlackHole 2ch (48 kHz)** — System Output routes to the virtual loopback device.
2. **AudioStream (left‑only)** — reads 1s float32 buffers from the input device.
3. **Resampler (→ 16 kHz)** — converts 48 kHz capture to 16 kHz arrays for Whisper.
4. **faster‑whisper** — produces segment JSONL with timestamps (`transcript.jsonl`). No audio is saved.
5. **Chunker → Embeddings → Chroma** — segment→chunks, sentence‑transformers embeddings, per‑session vector store.
6. **LM Studio** — map‑reduce **Summarize**, RAG‑style **Ask** using retrieved chunks.

---

## Install

1) **BlackHole 2ch** (loopback)
```bash
brew install blackhole-2ch
# System Settings → Sound → Output: choose "BlackHole 2ch"
# Audio MIDI Setup: set BlackHole 2ch *input & output* format to 48,000 Hz
# (Optional) Create a Multi‑Output Device so you can hear audio while recording
```

2) **Python env**
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

3) **LM Studio**
- Start the local server (e.g., `http://localhost:1234/v1`) with an instruction‑tuned model (e.g., `llama-3.1-8b-instruct`).

---

## Usage

### Start / Stop
```bash
# Use the exact input device name seen in `catchup devices`
catchup start --session "AI Day 2025" \
  --input-device "BlackHole 2ch" \
  --device cpu --compute-type int8 \
  --model small --lang en

# Later (new terminal)
catchup stop
```

> If the `catchup` entrypoint isn’t on PATH, run:  
> `python -m apps.cli.main start --session "AI Day 2025" --input-device "BlackHole 2ch"`

### Summarize
```bash
catchup summarize --session-id <id> --strategy map-reduce
# output: data/sessions/<id>/exports/summary.txt
```

### **Ask (RAG) – detailed**

Ask lets you query only what was captured in a session (no web search). It’s a **local RAG** pipeline:

```bash
catchup ask --session-id <id> "What were the main arguments about X?" --k 6
# output: data/sessions/<id>/exports/answer.txt
```

**What happens under the hood**
1. **Load** `transcript.jsonl` and **merge** segments into ~1.2–1.6k‑token chunks with ~10% rolling overlap.  
2. **Embed** each chunk using Sentence‑Transformers (default `all-MiniLM-L6-v2`, 384‑dim).  
3. **Upsert** vectors into **Chroma** at `data/sessions/<id>/chroma` (persisted between runs).  
4. **Query** Chroma with your question (top‑k = `--k`, default 6).  
5. **Prompt** LM Studio with a strict, sourced prompt (excerpts only) to draft an answer.  
6. **Write** the answer to `exports/answer.txt` (plain text).

**CLI options**
- `--k <int>`: number of chunks to retrieve (default **6**). Try **8–10** if answers seem incomplete.  
- `--out <path>`: write result to a custom file (default: `exports/answer.txt`).  
- `--session-id <id>`: session folder name (see `catchup list`).

**Config knobs (in `configs/default.yaml`)**
- `rag.top_k`: default top‑k for retrieval.  
- `rag.overlap_ratio`: rolling overlap between chunks (default ~0.1).  
- `rag.target_tokens`: desired chunk size in tokens (guides merging).  
- `rag.sim_threshold`: similarity cutoff (if used).

**Tuning tips**
- Use **`--model medium`** during capture for higher ASR quality.  
- If answers cite the wrong context, **increase `--k`** or re‑summarize for context first.  
- If you change embedding model or chunking logic, **rebuild** the vector store:
  ```bash
  catchup clean --session-id <id>
  # then run `catchup ask` again to re-embed
  ```

**Limitations (by design, v0)**
- Answers come **only** from the transcript; no web or external docs.  
- No “source snippets” in the output yet (planned: `--show-sources`).  
- Quality depends on capture clarity and chunking granularity.

### Export
```bash
catchup export --session-id <id> --formats txt,json,srt,vtt
# writes to data/sessions/<id>/exports/
```

### Devices helper
```bash
catchup devices
```

---

## Configuration

- Defaults in `configs/default.yaml` (overrides via `CATCHUP_*` env vars or CLI flags).

Key settings:
- **Audio:** capture 48 kHz, **left‑only** → resample to **16 kHz** for Whisper.
- **ASR:** faster‑whisper `small` (try `medium`), `--lang en|auto`.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384‑dim).
- **RAG:** per‑session Chroma store; `rag.top_k`, `rag.overlap_ratio`, `rag.target_tokens`.
- **LLM:** LM Studio (OpenAI‑compatible endpoint + local model).
  - Endpoint: `http://localhost:1234/v1`
  - Model: e.g., `llama-3.1-8b-instruct`
  - Temperature, max tokens are defined in `configs/default.yaml` under `llm`.

---

## Troubleshooting

- **Garbage transcript** → typically **sample rate** or **wrong BlackHole**:
  - System Output must be **BlackHole 2ch** (not 16ch).
  - Audio MIDI Setup: set BlackHole in/out to **48,000 Hz**.
  - Always pass `--input-device "BlackHole 2ch"`.
- **Silence** while playing audio → wrong input device or Chrome not routed.
  - Use `catchup devices` to confirm, or run a quick dBFS meter script.
- **Metal not recognized** by `ctranslate2` → use CPU for now:
  ```bash
  catchup start --session Test --device cpu --compute-type int8
  ```
  For Metal: prefer Python **3.12** and prebuilt wheels:
  ```bash
  pip uninstall -y ctranslate2 faster-whisper
  pip install --only-binary=:all: "ctranslate2>=4.0.0" "faster-whisper>=1.0.0"
  ```

---

## Roadmap & Improvements

### v1 — Cloud toggles (still CLI)
- Optional remote LLM/embeddings **if configured** (keep local‑only by default).
- Simple `.env` support to point at a remote endpoint.

### v2 — Streamlit UI
- Start/Stop buttons, live transcript tail, configurable model/lang, one‑click **Summarize**/**Ask**/**Export**.
- Minimal session browser (open previous sessions).

### v3 — Visuals (screenshots)
- Screen Recording permission & strategic **WebP** screenshots (e.g., on phrases: “as you can see…”, “this slide…”).
- Associate shots with nearest transcript timestamp.
- v3.1: OCR on screenshots (Tesseract / PaddleOCR) → searchable with transcript.

### RAG quality upgrades
- **MMR** diversification for retrieval; **cross‑encoder rerank** for precision (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).  
- Incremental embeddings cache; avoid re‑embedding unchanged chunks.  
- Topic‑aware chunking (rolling similarity, silences, cue phrases).  
- Structured outputs (JSON) for summaries; follow‑up prompt templates.

### Audio/MIDI ergonomics (macOS)
- Provide a `catchup route` command using `SwitchAudioSource` to toggle system output:
  ```bash
  brew install switchaudio-osx
  SwitchAudioSource -t output -s "BlackHole 2ch"
  # restore later:
  SwitchAudioSource -t output -s "MacBook Air Speakers"
  ```
- Optional helper to create a **Multi‑Output Device** programmatically (via AppleScript / `SwitchAudioSource`), so users always hear audio while capturing.

### Reliability & Ops
- Low‑disk watchdog with cutoff parameter; graceful stop when threshold hit.
- Backpressure/overflow logging; warning when sustained clipping/silence.
- Optional VAD to skip long silences; optional word‑timestamps mode.

### Packaging & DX
- `pipx` / Homebrew formula, `catchup route` + `catchup doctor` (self‑tests for device & sample‑rate).
- Minimal CI for lint/type checks and README link health.

---

## Disclaimer

CatchUp runs locally and stores only text. Use responsibly and only where you have rights.
