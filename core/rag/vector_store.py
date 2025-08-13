# core/rag/vector_store.py
from __future__ import annotations
import json
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

# A lightweight chunk record interface expected by this store
# (merge_segments produces these fields)
class _Chunk:
    def __init__(self, chunk_id: str, text: str, t_start: float, t_end: float, seg_ids: list[str] | None = None):
        self.chunk_id = chunk_id
        self.text = text
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.seg_ids = seg_ids or []


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce metadata to Chroma-compatible primitives."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            # Store as compact JSON string
            out[f"{k}_json"] = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
            # Optionally add helpful scalar summaries
            if isinstance(v, list):
                out[f"{k}_count"] = len(v)
        else:
            out[k] = str(v)
    return out


class ChromaStore:
    def __init__(self, persist_directory: str, collection_name: str = "chunks"):
        # Using default settings; cosine space is typical for ST embeddings
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: List[_Chunk], embeddings: List[list[float]]):
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[List[float]] = []

        for ch, emb in zip(chunks, embeddings):
            cid = ch.chunk_id or f"chunk-{len(ids):06d}"
            ids.append(cid)
            docs.append(ch.text)

            meta = {
                "t_start": float(ch.t_start),
                "t_end": float(ch.t_end),
                "char_len": len(ch.text),
            }
            # seg_ids may be a list → stringify + count
            if getattr(ch, "seg_ids", None):
                meta["seg_ids"] = ch.seg_ids  # will become seg_ids_json + seg_ids_count
            metas.append(_sanitize_metadata(meta))

            # ensure plain python floats
            embs.append([float(x) for x in (emb.tolist() if hasattr(emb, "tolist") else emb)])

        self.collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embs,
            metadatas=metas,
        )
        return len(ids)

    def query(self, query: str, k: int = 6):
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        # Chroma returns lists-of-lists
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # Try to expose seg_ids if present as parsed list (best effort)
            seg_ids = None
            if "seg_ids_json" in meta:
                try:
                    seg_ids = json.loads(meta["seg_ids_json"])
                except Exception:
                    seg_ids = None
            hits.append({
                "text": doc,
                "distance": float(dist) if dist is not None else None,
                "t_start": float(meta.get("t_start", 0.0)),
                "t_end": float(meta.get("t_end", 0.0)),
                "seg_ids": seg_ids,
            })
        return hits
