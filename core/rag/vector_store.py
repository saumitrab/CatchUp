from __future__ import annotations
import chromadb
from chromadb.config import Settings
from typing import List, Dict

class ChromaStore:
    def __init__(self, persist_directory: str, collection: str = "catchup"):
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection)

    def reset(self):
        self.client.reset()

    def upsert(self, chunks: list, embeddings):
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [dict(seg_ids=c.seg_ids, t_start=c.t_start, t_end=c.t_end, token_count=c.token_count) for c in chunks]
        self.collection.upsert(ids=ids, embeddings=embeddings.tolist(), documents=docs, metadatas=metas)

    def query(self, query: str, k: int = 6):
        res = self.collection.query(query_texts=[query], n_results=k, include=["metadatas", "documents", "embeddings", "distances"])
        out = []
        if res and res["ids"]:
            for i in range(len(res["ids"][0])):
                out.append({
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "meta": res["metadatas"][0][i],
                    "distance": res["distances"][0][i] if res.get("distances") else None,
                })
        return out
