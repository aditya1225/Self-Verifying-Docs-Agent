"""Parallel hybrid retrieval (Ch. 3).

BM25, vector search, and every sub-question's retrieval are launched
concurrently, fused with RRF, then re-ranked once over the merged candidate
pool. Running the re-ranker once on the fused pool - rather than per
sub-question - is what keeps multi-hop latency close to single-hop.
"""

from __future__ import annotations

import asyncio

from ..config import get_settings
from ..index.store import CorpusIndex
from ..reliability import Degradation, with_timeout
from ..schemas import RetrievalResult, ScoredChunk
from ..tracing import Tracer
from .fusion import reciprocal_rank_fusion
from .rerank import RerankerUnavailable, fallback_rerank, rerank
from .retrievers import bm25_search, vector_search

# Relative RRF weight of a planner sub-question against the user's own query.
SUB_QUESTION_WEIGHT = 0.6


async def hybrid_retrieve(
    index: CorpusIndex,
    queries: list[str],
    *,
    tracer: Tracer,
    use_vector: bool = True,
    use_bm25: bool = True,
    use_rerank: bool = True,
    degradations: list[Degradation] | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    settings = get_settings()
    degradations = degradations if degradations is not None else []
    queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
    if not queries:
        return RetrievalResult(query="", chunks=[])

    primary = queries[0]
    top_k = top_k or settings.rerank_top_k

    # --- fan out ---------------------------------------------------------
    # Each job is (retriever name, is_primary_query, future).
    jobs: list[tuple[str, bool, asyncio.Future]] = []
    with tracer.span("retrieval.parallel", queries=queries, bm25=use_bm25, vector=use_vector):
        for query in queries:
            is_primary = query == primary
            if use_bm25:
                jobs.append(
                    (
                        "bm25",
                        is_primary,
                        asyncio.ensure_future(
                            with_timeout(
                                asyncio.to_thread(bm25_search, index, query, settings.bm25_top_k),
                                settings.retrieval_timeout_s,
                                component="bm25",
                                fallback=[],
                                degradations=degradations,
                            )
                        ),
                    )
                )
            if use_vector:
                jobs.append(
                    (
                        "vector",
                        is_primary,
                        asyncio.ensure_future(
                            with_timeout(
                                asyncio.to_thread(vector_search, index, query, settings.vector_top_k),
                                settings.retrieval_timeout_s,
                                component="vector",
                                fallback=[],
                                degradations=degradations,
                            )
                        ),
                    )
                )

        results: list[list[ScoredChunk]] = list(await asyncio.gather(*(job[2] for job in jobs)))

    per_retriever: dict[str, int] = {}
    for (name, _, _), ranked in zip(jobs, results):
        per_retriever[name] = per_retriever.get(name, 0) + len(ranked)

    notes: list[str] = []
    degraded = False
    if use_bm25 and per_retriever.get("bm25", 0) == 0:
        notes.append("bm25 returned nothing")
    if use_vector and per_retriever.get("vector", 0) == 0:
        notes.append("vector search returned nothing")
        degraded = True

    # --- fuse ------------------------------------------------------------
    # Sub-question lists are down-weighted so a tangent in one hop cannot
    # outvote the question the user actually asked.
    non_empty: list[list[ScoredChunk]] = []
    weights: list[float] = []
    for (_, is_primary, _), ranked in zip(jobs, results):
        if not ranked:
            continue
        non_empty.append(ranked)
        weights.append(1.0 if is_primary else SUB_QUESTION_WEIGHT)

    if not non_empty:
        return RetrievalResult(
            query=primary,
            chunks=[],
            per_retriever_counts=per_retriever,
            degraded=True,
            notes=notes + ["all retrievers returned empty"],
        )

    with tracer.span("retrieval.fusion", lists=len(non_empty)):
        fused = reciprocal_rank_fusion(
            non_empty,
            k=settings.rrf_k,
            weights=weights,
            top_k=max(top_k * 3, settings.rerank_top_k * 3),
        )

    # --- re-rank ---------------------------------------------------------
    reranked_ok = False
    if use_rerank and fused:
        with tracer.span("retrieval.rerank", candidates=len(fused)) as span:
            try:
                final = await asyncio.wait_for(
                    asyncio.to_thread(rerank, primary, fused, top_k),
                    timeout=settings.rerank_timeout_s,
                )
                reranked_ok = True
                span.attributes["backend"] = "cross-encoder"
            except (RerankerUnavailable, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                degradations.append(Degradation("reranker", f"{type(exc).__name__}: {exc}"))
                notes.append("re-ranker unavailable; kept reciprocal-rank-fusion order")
                degraded = True
                final = fallback_rerank(fused, top_k)
                span.attributes["backend"] = "rrf-fallback"
    else:
        final = fallback_rerank(fused, top_k)

    return RetrievalResult(
        query=primary,
        chunks=final,
        per_retriever_counts=per_retriever,
        reranked=reranked_ok,
        degraded=degraded,
        notes=notes,
    )
