from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re, math

@dataclass
class Seg:
    seg_id: str
    t_start: float
    t_end: float
    text: str

@dataclass
class Chunk:
    chunk_id: str
    seg_ids: list[str]
    t_start: float
    t_end: float
    text: str
    token_count: int

def rough_token_count(s: str) -> int:
    # very rough approximation
    return max(1, len(re.findall(r"\w+|\S", s)) // 1)

def merge_segments(segments: List[Seg], target_tokens=1400, overlap_ratio=0.1) -> List[Chunk]:
    chunks: List[Chunk] = []
    cur_text, cur_ids = [], []
    cur_start, cur_end = None, None
    token_sum = 0
    idx = 0
    for seg in segments:
        t = seg.text.strip()
        tc = rough_token_count(t)
        if cur_start is None:
            cur_start = seg.t_start
        if token_sum + tc > target_tokens and cur_text:
            # flush current chunk
            text = " ".join(cur_text).strip()
            chunks.append(Chunk(
                chunk_id=f"chk-{len(chunks):06d}",
                seg_ids=cur_ids[:],
                t_start=cur_start,
                t_end=cur_end if cur_end is not None else cur_start,
                text=text,
                token_count=rough_token_count(text),
            ))
            # start new with overlap
            overlap_n = max(1, int(len(cur_ids) * overlap_ratio))
            cur_ids = cur_ids[-overlap_n:]
            cur_text = cur_text[-overlap_n:]
            token_sum = rough_token_count(" ".join(cur_text))
            cur_start = segments[idx - overlap_n + 1].t_start if (idx - overlap_n + 1) >= 0 else seg.t_start
        # add seg
        cur_ids.append(seg.seg_id)
        cur_text.append(t)
        token_sum += tc
        cur_end = seg.t_end
        idx += 1
    if cur_text:
        text = " ".join(cur_text).strip()
        chunks.append(Chunk(
            chunk_id=f"chk-{len(chunks):06d}",
            seg_ids=cur_ids[:],
            t_start=cur_start if cur_start is not None else 0.0,
            t_end=cur_end if cur_end is not None else cur_start or 0.0,
            text=text,
            token_count=rough_token_count(text),
        ))
    return chunks
