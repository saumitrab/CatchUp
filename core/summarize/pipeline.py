from __future__ import annotations
from typing import List
from dataclasses import dataclass

MAP_PROMPT = (
    "You are a diligent colleague who attended this talk. "
    "Write concise bullet points of the key claims, evidence/examples, definitions, and action items in this chunk. "
    "Be factual. Do not hallucinate."
)
REDUCE_PROMPT = (
    "Combine these chunk summaries into a cohesive document with sections: "
    "TL;DR, Key Points, Examples/Case Studies, Takeaways/Actions, Open Questions, Suggested Follow-ups. "
    "Use only the provided content; no external info."
)

def map_reduce(chunks: List[str], llm) -> str:
    # chunks: list of chunk text
    maps = []
    for ch in chunks:
        maps.append(llm.chat([
            {"role":"system","content":MAP_PROMPT},
            {"role":"user","content":ch},
        ]))
    combined = "\n\n".join(maps)
    final = llm.chat([
        {"role":"system","content":REDUCE_PROMPT},
        {"role":"user","content":combined},
    ])
    return final

def simple(transcript_text: str, llm) -> str:
    prompt = (
        "Summarize the following transcript into bullet points with: TL;DR, Key Points, Takeaways/Actions, Suggested Follow-ups. "
        "Do not add external information.\n\n" + transcript_text
    )
    return llm.chat([{"role":"user","content":prompt}])
