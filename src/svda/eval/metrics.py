"""Metrics (Ch. 19).

One rule governs this module: **the evaluator never trusts the pipeline's own
critic**. If faithfulness were read off `answer.critic`, the `naive` profile
(which has no critic) would score nothing and the `full` profile would be
grading its own homework. So faithfulness is judged here, independently, by a
separate critic instance against the chunks the pipeline actually retrieved.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Sequence

from ..index.store import CorpusIndex
from ..llm.stub import content_terms
from ..schemas import (
    AgentAnswer,
    AnswerStatus,
    GoldenItem,
    ItemResult,
    ProfileMetrics,
    Route,
)

ANSWER_KEYWORD_THRESHOLD = 0.6


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def keyword_hit_rate(text: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    haystack = _normalize(text)
    return sum(1 for kw in keywords if _normalize(kw) in haystack) / len(keywords)


def answer_is_correct(item: GoldenItem, answer: AgentAnswer) -> bool:
    if not item.answerable:
        return answer.status in {AnswerStatus.DECLINED, AnswerStatus.BLOCKED}
    if answer.status is not AnswerStatus.ANSWERED:
        return False
    return keyword_hit_rate(answer.answer, item.answer_keywords) >= ANSWER_KEYWORD_THRESHOLD


def retrieval_scores(
    item: GoldenItem, answer: AgentAnswer, index: CorpusIndex
) -> tuple[float, float, bool]:
    """(recall, precision, chapter_hit) against the item's context keywords.

    Keyword-based rather than chunk-id based so the metric survives a re-chunk
    - pinning to ids would make every chunking change look like a regression.
    """
    chunks = [c for c in (index.get(cid) for cid in answer.retrieved_chunk_ids) if c is not None]
    if not chunks:
        return 0.0, 0.0, False

    texts = [_normalize(c.text) for c in chunks]
    joined = " ".join(texts)

    keywords = [_normalize(k) for k in item.context_keywords]
    recall = (
        sum(1 for kw in keywords if kw in joined) / len(keywords) if keywords else 1.0
    )
    if keywords:
        relevant = sum(1 for text in texts if any(kw in text for kw in keywords))
        precision = relevant / len(texts)
    else:
        precision = 1.0

    chapter_hit = (
        any(c.chapter_number in item.expected_chapters for c in chunks)
        if item.expected_chapters
        else True
    )
    return recall, precision, chapter_hit


def citation_coverage(answer: AgentAnswer) -> float:
    """Fraction of factual sentences in the final text carrying a marker."""
    if answer.status is not AnswerStatus.ANSWERED:
        return 1.0 if answer.is_refusal else 0.0
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.answer.strip()) if len(s.strip()) > 25]
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if re.search(r"\[\d+\]", s)) / len(sentences)


def extractive_faithfulness(answer: AgentAnswer, index: CorpusIndex) -> float:
    """Offline grounding check used when no judge LLM is configured.

    For each sentence, what fraction of its content terms appear in the chunks
    it cites? A sentence is faithful if that fraction clears 0.85.
    """
    if answer.status is not AnswerStatus.ANSWERED:
        return 1.0 if answer.is_refusal else 0.0

    by_marker = {c.marker: c.chunk_id for c in answer.citations}
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.answer.strip()) if s.strip()]
    if not sentences:
        return 0.0

    scored = 0
    supported = 0
    for sentence in sentences:
        markers = re.findall(r"\[\d+\]", sentence)
        body = re.sub(r"\[\d+\]", " ", sentence)
        terms = content_terms(body)
        if len(terms) < 3:
            continue
        scored += 1
        if not markers:
            continue
        haystack = " ".join(
            (index.get(by_marker[m]).text.lower() if by_marker.get(m) and index.get(by_marker[m]) else "")
            for m in markers
        )
        if not haystack:
            continue
        covered = sum(1 for t in terms if t in haystack)
        if covered / len(terms) >= 0.85:
            supported += 1

    return supported / scored if scored else 0.0


def route_is_correct(item: GoldenItem, answer: AgentAnswer) -> bool:
    if answer.route is None:
        return False
    if item.kind is Route.MULTI_HOP:
        # Treating a multi-hop question as simple is the failure that matters;
        # the reverse only costs money.
        return answer.route.route is Route.MULTI_HOP
    return answer.route.route is item.kind


def score_item(
    item: GoldenItem,
    answer: AgentAnswer,
    index: CorpusIndex,
    *,
    judged_faithfulness: float | None = None,
) -> ItemResult:
    recall, precision, chapter_hit = retrieval_scores(item, answer, index)
    faith = (
        judged_faithfulness
        if judged_faithfulness is not None
        else extractive_faithfulness(answer, index)
    )
    coverage = citation_coverage(answer)
    return ItemResult(
        item_id=item.id,
        question=item.question,
        kind=item.kind,
        answerable=item.answerable,
        status=answer.status,
        answer=answer.answer,
        correct=answer_is_correct(item, answer),
        faithful=faith >= 0.8,
        faithfulness=faith,
        citation_coverage=coverage,
        retrieval_recall=recall,
        retrieval_precision=precision,
        chapter_hit=chapter_hit,
        refused_correctly=(not item.answerable) and answer.is_refusal,
        route_correct=route_is_correct(item, answer),
        latency_ms=answer.latency_ms,
        cost_usd=answer.usage.cost_usd,
    )


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank: the smallest value at or above the given percentile.
    index = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return float(ordered[index])


def aggregate(profile: str, results: list[ItemResult]) -> ProfileMetrics:
    if not results:
        return ProfileMetrics(profile=profile)

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    multihop = [r for r in answerable if r.kind is Route.MULTI_HOP]
    # Faithfulness and citation coverage are only meaningful where the agent
    # produced an answer *from retrieved sources*. A chit-chat reply is a
    # canned sentence that makes no factual claim, so scoring it for citations
    # would quietly drag the metric down for doing the right thing.
    answered = [
        r
        for r in results
        if r.status is AnswerStatus.ANSWERED and r.kind is not Route.CHITCHAT
    ]
    latencies = [r.latency_ms for r in results]

    return ProfileMetrics(
        profile=profile,
        n=len(results),
        answer_accuracy=_mean([1.0 if r.correct else 0.0 for r in answerable]),
        faithfulness=_mean([r.faithfulness for r in answered]),
        citation_coverage=_mean([r.citation_coverage for r in answered]),
        retrieval_recall=_mean([r.retrieval_recall for r in answerable]),
        retrieval_precision=_mean([r.retrieval_precision for r in answerable]),
        multihop_accuracy=_mean([1.0 if r.correct else 0.0 for r in multihop]),
        refusal_correctness=_mean([1.0 if r.refused_correctly else 0.0 for r in unanswerable]),
        route_accuracy=_mean([1.0 if r.route_correct else 0.0 for r in results]),
        mean_latency_ms=_mean(latencies),
        p95_latency_ms=percentile(latencies, 95),
        mean_cost_usd=_mean([r.cost_usd for r in results]),
        total_cost_usd=sum(r.cost_usd for r in results),
        errors=sum(1 for r in results if r.error),
        extra={
            "n_answered": len(answered),
            "n_refused": sum(
                1 for r in results if r.status in {AnswerStatus.DECLINED, AnswerStatus.BLOCKED}
            ),
            "n_human_review": sum(1 for r in results if r.status is AnswerStatus.HUMAN_REVIEW),
            "n_multihop": len(multihop),
            "n_unanswerable": len(unanswerable),
        },
    )
