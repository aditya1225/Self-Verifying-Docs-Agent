"""Cross-encoder re-ranking, with the fusion score as the designed fallback.

If the cross-encoder is unavailable or slow, we keep the RRF order instead of
failing the request - and we say so in the trace (Ch. 12).
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from ..schemas import ScoredChunk


class RerankerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def cross_encoder_available() -> bool:
    try:
        _load_cross_encoder()
        return True
    except Exception:  # noqa: BLE001
        return False


def rerank(query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Raises RerankerUnavailable so the caller can record the degradation."""
    if not candidates:
        return []
    try:
        model = _load_cross_encoder()
        scores = model.predict(
            [(query, sc.chunk.text) for sc in candidates],
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RerankerUnavailable(str(exc)) from exc

    reranked = [
        ScoredChunk(chunk=sc.chunk, score=float(score), retriever=f"{sc.retriever}+rerank", rank=0)
        for sc, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda sc: (-sc.score, sc.chunk.id))
    for rank, sc in enumerate(reranked, start=1):
        sc.rank = rank
    return reranked[:top_k]


def fallback_rerank(candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Keep fusion order - the documented degradation path."""
    return candidates[:top_k]
