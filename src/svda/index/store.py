"""The corpus index: chunks + BM25 + a dense matrix, persisted to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..reliability import CorpusNotIndexedError
from ..schemas import Chunk
from .embeddings import Embedder, build_embedder

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


def bm25_tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9][a-z0-9\-']+", text.lower())


@dataclass
class IndexMeta:
    doc_id: str
    source_path: str
    embedding_backend: str
    embedding_dim: int
    n_chunks: int
    created_at: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class CorpusIndex:
    """Holds everything retrieval needs. Built once, then loaded read-only."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: IndexMeta) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
        self.chunks = chunks
        self.vectors = vectors
        self.meta = meta
        self._by_id = {chunk.id: chunk for chunk in chunks}
        self._bm25 = None
        self._embedder: Embedder | None = None
        self._vocabulary: set[str] | None = None

    # -- accessors ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([bm25_tokenize(c.text) for c in self.chunks])
        return self._bm25

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            backend = "hash" if self.meta.embedding_backend.startswith("hash") else "local"
            embedder = build_embedder(backend)
            if embedder.dim != self.meta.embedding_dim:
                raise CorpusNotIndexedError(
                    f"index was built with {self.meta.embedding_backend} "
                    f"(dim {self.meta.embedding_dim}) but the available embedder is "
                    f"{embedder.name} (dim {embedder.dim}). Re-run `svda ingest`."
                )
            self._embedder = embedder
        return self._embedder

    @property
    def vocabulary(self) -> set[str]:
        """Corpus term set, used by the router's out-of-scope check."""
        if self._vocabulary is None:
            vocab: set[str] = set()
            for chunk in self.chunks:
                vocab.update(bm25_tokenize(chunk.text))
            self._vocabulary = vocab
        return self._vocabulary

    # -- persistence -------------------------------------------------------
    def save(self, directory: Path | None = None) -> Path:
        directory = directory or get_settings().index_dir
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        np.save(directory / VECTORS_FILE, self.vectors)
        (directory / META_FILE).write_text(json.dumps(self.meta.to_dict(), indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "CorpusIndex":
        directory = directory or get_settings().index_dir
        chunks_path = directory / CHUNKS_FILE
        if not chunks_path.exists():
            raise CorpusNotIndexedError(
                f"no index at {directory}. Run: svda ingest --pdf <path-to-pdf>"
            )
        chunks = [
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(directory / VECTORS_FILE)
        meta = IndexMeta(**json.loads((directory / META_FILE).read_text(encoding="utf-8")))
        return cls(chunks, vectors, meta)
