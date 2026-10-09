"""Embedding backends.

`local` uses a sentence-transformers bi-encoder. `hash` is a dependency-free
deterministic fallback (hashed word + character n-grams) so ingest, tests and
CI work on a machine without torch. The backend name is persisted with the
index so a query never gets embedded by a different model than the corpus.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from ..config import get_settings

HASH_DIM = 768


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashEmbedder:
    """Deterministic bag-of-features embedding. No model download, no network."""

    def __init__(self, dim: int = HASH_DIM) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            words = [w for w in lowered.split() if w]
            features: list[str] = []
            features.extend(words)
            features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
            for word in words:
                padded = f"#{word}#"
                features.extend(padded[i : i + 4] for i in range(max(len(padded) - 3, 0)))
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, bucket] += sign
        # Sub-linear damping keeps long chunks from dominating cosine scores.
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(backend: str | None = None) -> Embedder:
    settings = get_settings()
    backend = (backend or settings.embedding_backend).lower()
    if backend == "hash":
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except Exception:  # noqa: BLE001 - torch missing or model not downloadable
        return HashEmbedder()
