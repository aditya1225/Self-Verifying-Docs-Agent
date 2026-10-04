"""Reciprocal rank fusion.

RRF combines ranked lists without needing the scores to be comparable, which
is the whole problem with mixing BM25 scores and cosine similarities.
"""

from __future__ import annotations

from ..schemas import ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights and ranked_lists must be the same length")

    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    sources: dict[str, set[str]] = {}

    for weight, ranked in zip(weights, ranked_lists):
        for rank, scored in enumerate(ranked, start=1):
            chunk_id = scored.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            sources.setdefault(chunk_id, set()).add(scored.retriever)
            if chunk_id not in best or rank < best[chunk_id].rank:
                best[chunk_id] = scored

    fused = [
        ScoredChunk(
            chunk=best[chunk_id].chunk,
            score=score,
            retriever="+".join(sorted(sources[chunk_id])),
            rank=0,
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda sc: (-sc.score, sc.chunk.id))
    for rank, scored in enumerate(fused, start=1):
        scored.rank = rank
    return fused[:top_k] if top_k else fused
