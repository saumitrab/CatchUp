from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np, torch

class STEncoder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        self.model = SentenceTransformer(model, device=device)
    def embed_texts(self, texts: list[str]) -> "np.ndarray":
        embs = self.model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        return embs
