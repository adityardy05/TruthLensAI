"""Embedding wrapper around all-MiniLM-L6-v2 (an English-centric model).

The embedder is created once and shared. It returns L2-normalised float32
vectors so a dot product equals cosine similarity, which is what the
``IndexFlatIP`` FAISS index expects. No fallback model exists on purpose:
mixing embedding spaces in one index silently corrupts similarity scores.
"""
from __future__ import annotations

import logging
import threading
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model_name: str
    dim: int

    @property
    def available(self) -> bool: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, eager: bool = True):
        self.model_name = model_name
        self.dim = EMBED_DIM
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()
        if eager:
            self.load()

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model %s", self.model_name)
                return True
            except Exception as exc:  # missing package, offline, corrupt cache...
                self._load_failed = True
                logger.error("Embedding model %s unavailable: %s", self.model_name, exc)
                return False

    @property
    def available(self) -> bool:
        return self._model is not None or (not self._load_failed and self.load())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if not self.available:
            raise EmbeddingError(f"Embedding model {self.model_name} is not available")
        try:
            with self._lock:
                vectors = self._model.encode(
                    list(texts), batch_size=32, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
