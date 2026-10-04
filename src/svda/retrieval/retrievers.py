"""The two base retrievers. Both are sync and CPU-bound; the hybrid layer runs
them off the event loop in threads so they overlap (Ch. 3)."""

from __future__ import annotations

import numpy as np

from ..index.store import CorpusIndex, bm25_tokenize
from ..schemas import ScoredChunk


def bm25_search(index: CorpusIndex, query: str, top_k: int) -> list[ScoredChunk]:
    if len(index) == 0:
        return []
    tokens = bm25_tokenize(query)
    if not tokens:
        return []
    scores = np.asarray(index.bm25.get_scores(tokens), dtype=np.float32)
    return _top(index, scores, top_k, retriever="bm25", drop_nonpositive=True)


def vector_search(index: CorpusIndex, query: str, top_k: int) -> list[ScoredChunk]:
    if len(index) == 0:
        return []
    query_vector = index.embedder.encode([query])[0]
    # Both sides are L2-normalised, so the dot product is cosine similarity.
    scores = index.vectors @ query_vector.astype(np.float32)
    return _top(index, scores, top_k, retriever="vector", drop_nonpositive=False)


def _top(
    index: CorpusIndex,
    scores: np.ndarray,
    top_k: int,
    *,
    retriever: str,
    drop_nonpositive: bool,
) -> list[ScoredChunk]:
    if scores.size == 0:
        return []
    k = min(top_k, scores.size)
    # argpartition is O(n); we only sort the k we keep.
    candidates = np.argpartition(-scores, k - 1)[:k]
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")]

    out: list[ScoredChunk] = []
    for rank, idx in enumerate(candidates, start=1):
        score = float(scores[idx])
        if drop_nonpositive and score <= 0:
            continue
        out.append(
            ScoredChunk(chunk=index.chunks[int(idx)], score=score, retriever=retriever, rank=rank)
        )
    return out
