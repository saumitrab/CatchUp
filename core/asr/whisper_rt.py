from __future__ import annotations
import numpy as np, time, logging
from faster_whisper import WhisperModel
from dataclasses import dataclass

logger = logging.getLogger("catchup")

@dataclass
class Segment:
    seg_id: str
    t_start: float
    t_end: float
    text: str
    avg_logprob: float

class WhisperRealtime:
    def __init__(self, model_name="small", device="auto", compute_type="int8_float16", language="en", beam_size=1):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe_chunk(self, audio_np: np.ndarray, offset_sec: float):
        # Non-streaming: we run transcribe on each chunk and offset timestamps.
        # This is a pragmatic MVP approach.
        segments, info = self.model.transcribe(
            audio=audio_np,
            language=self.language,
            beam_size=self.beam_size,
        )
        for i, seg in enumerate(segments):
            yield Segment(
                seg_id=f"seg-{int(offset_sec*1000):06d}-{i:03d}",
                t_start=offset_sec + float(seg.start or 0.0),
                t_end=offset_sec + float(seg.end or 0.0),
                text=seg.text.strip(),
                avg_logprob=float(getattr(seg, "avg_logprob", -0.0)),
            )
