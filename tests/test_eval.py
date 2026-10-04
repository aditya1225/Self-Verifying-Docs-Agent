"""Evaluation harness, CI gate and the golden set itself."""

from __future__ import annotations

import pytest

from svda.eval import ci_gate
from svda.eval.golden import load_golden, summarize
from svda.eval.metrics import aggregate, citation_coverage, keyword_hit_rate, score_item
from svda.eval.report import ablation_markdown, render_dashboard
from svda.eval.runner import run_ablation, save_report
from svda.schemas import (
    AgentAnswer,
    AnswerStatus,
    Citation,
    EvalReport,
    GoldenItem,
    ItemResult,
    ProfileMetrics,
    Route,
)


# -- the golden set ---------------------------------------------------------


def test_golden_set_loads_and_is_well_formed():
    items = load_golden()
    assert len(items) >= 50, "the brief asks for 50-200 golden Q/A pairs"
    assert len({i.id for i in items}) == len(items)


def test_golden_set_covers_every_route_and_both_answerabilities():
    counts = summarize(load_golden())
    for kind in (Route.SIMPLE, Route.MULTI_HOP, Route.OUT_OF_SCOPE, Route.CHITCHAT):
        assert counts.get(kind.value, 0) > 0, f"no golden items of kind {kind.value}"
    assert counts["unanswerable"] >= 3, "need refusal cases to measure refusal correctness"


def test_golden_set_spans_most_of_the_book():
    chapters = {c for item in load_golden() for c in item.expected_chapters}
    assert len(chapters) >= 10


# -- metrics ----------------------------------------------------------------


def test_keyword_hit_rate():
    assert keyword_hit_rate("The leader accepts all writes.", ["leader", "writes"]) == 1.0
    assert keyword_hit_rate("The leader accepts all writes.", ["leader", "quorum"]) == 0.5
    assert keyword_hit_rate("anything", []) == 1.0


def test_citation_coverage_counts_factual_sentences_only():
    answered = AgentAnswer(
        query="q",
        status=AnswerStatus.ANSWERED,
        answer="Replication keeps copies of data on several machines [1]. Ok.",
    )
    assert citation_coverage(answered) == 1.0

    uncited = AgentAnswer(
        query="q",
        status=AnswerStatus.ANSWERED,
        answer="Replication keeps copies of data on several machines without any source.",
    )
    assert citation_coverage(uncited) == 0.0


def test_refusal_scores_as_correct_for_unanswerable_items(index):
    item = GoldenItem(id="u", question="q", answerable=False)
    refusal = AgentAnswer(query="q", status=AnswerStatus.DECLINED, answer="I can't support this.")
    result = score_item(item, refusal, index)
    assert result.correct and result.refused_correctly


def test_answering_an_unanswerable_item_scores_as_wrong(index):
    item = GoldenItem(id="u", question="q", answerable=False)
    hallucination = AgentAnswer(
        query="q",
        status=AnswerStatus.ANSWERED,
        answer="Revenue was $400m last year. [1]",
        citations=[
            Citation(marker="[1]", chunk_id="test::c00001", label="l", chapter="c",
                     section="s", page_start=1, page_end=2)
        ],
    )
    result = score_item(item, hallucination, index)
    assert not result.correct and not result.refused_correctly


def test_aggregate_separates_answerable_from_refusal_populations():
    results = [
        ItemResult(item_id="a", question="q", kind=Route.SIMPLE, answerable=True,
                   status=AnswerStatus.ANSWERED, answer="x", correct=True,
                   faithfulness=1.0, citation_coverage=1.0, latency_ms=100),
        ItemResult(item_id="b", question="q", kind=Route.MULTI_HOP, answerable=True,
                   status=AnswerStatus.ANSWERED, answer="x", correct=False,
                   faithfulness=0.5, citation_coverage=1.0, latency_ms=300),
        ItemResult(item_id="c", question="q", kind=Route.SIMPLE, answerable=False,
                   status=AnswerStatus.DECLINED, answer="", correct=True,
                   refused_correctly=True, latency_ms=50),
    ]
    metrics = aggregate("full", results)
    assert metrics.n == 3
    assert metrics.answer_accuracy == pytest.approx(0.5)  # 1 of 2 answerable
    assert metrics.multihop_accuracy == pytest.approx(0.0)
    assert metrics.refusal_correctness == pytest.approx(1.0)
    assert metrics.faithfulness == pytest.approx(0.75)  # answered items only
    assert metrics.mean_latency_ms == pytest.approx(150.0)


# -- CI gate ----------------------------------------------------------------


def _report(**overrides) -> EvalReport:
    base = dict(
        profile="full", n=10, faithfulness=0.9, citation_coverage=1.0,
        answer_accuracy=0.8, refusal_correctness=1.0, retrieval_recall=0.8,
        p95_latency_ms=500.0,
    )
    base.update(overrides)
    return EvalReport(
        run_id="r", created_at="now", llm_backend="stub",
        profiles=[ProfileMetrics(**base)],
    )


def test_gate_passes_a_healthy_report(tmp_path):
    result = ci_gate.check(_report(), baseline_path=tmp_path / "missing.json")
    assert result.passed and not result.baseline_found


def test_gate_fails_on_an_absolute_floor(tmp_path):
    result = ci_gate.check(_report(faithfulness=0.4), baseline_path=tmp_path / "missing.json")
    assert not result.passed
    assert any(f.metric == "faithfulness" and f.kind == "floor" for f in result.findings)


def test_gate_fails_on_a_latency_ceiling(tmp_path):
    result = ci_gate.check(_report(p95_latency_ms=120_000.0), baseline_path=tmp_path / "missing.json")
    assert not result.passed
    assert any(f.kind == "ceiling" for f in result.findings)


def test_gate_fails_on_regression_against_the_baseline(tmp_path):
    baseline = ProfileMetrics(
        profile="full", faithfulness=0.95, citation_coverage=1.0, answer_accuracy=0.85,
        refusal_correctness=1.0, retrieval_recall=0.8,
    )
    # 0.80 is above the 0.70 floor but 15 points below the baseline.
    result = ci_gate.check(_report(faithfulness=0.80), baseline=baseline)
    assert not result.passed
    assert any(f.kind == "regression" for f in result.findings)


def test_gate_tolerates_small_noise():
    baseline = ProfileMetrics(
        profile="full", faithfulness=0.90, citation_coverage=1.0, answer_accuracy=0.80,
        refusal_correctness=1.0, retrieval_recall=0.80,
    )
    result = ci_gate.check(_report(faithfulness=0.88), baseline=baseline)
    assert result.passed


# -- end-to-end ablation ----------------------------------------------------


async def test_ablation_runs_every_profile_and_renders(index, tmp_settings):
    report = await run_ablation(index, limit=12, concurrency=4)

    assert [p.profile for p in report.profiles] == ["naive", "hybrid", "agentic", "full"]
    for metrics in report.profiles:
        assert metrics.n == 12
        assert metrics.errors == 0, "no golden item should raise"

    markdown = ablation_markdown(report)
    assert "Faithfulness" in markdown and "Naive RAG" in markdown

    html = render_dashboard(report, spans=[])
    assert "<title>" in html and "Ablation" in html

    path = save_report(report, tmp_settings.runs_dir)
    assert path.exists()


async def test_reflection_does_not_reduce_faithfulness(index, tmp_settings):
    """The whole premise of the project: the critic should not make grounding worse."""
    report = await run_ablation(index, profiles=["agentic", "full"], limit=16, concurrency=4)
    before = report.metrics_for("agentic")
    after = report.metrics_for("full")
    assert after.faithfulness >= before.faithfulness - 1e-9


async def test_full_profile_refuses_the_unanswerable_items(index, tmp_settings):
    report = await run_ablation(index, profiles=["full"], concurrency=4)
    metrics = report.metrics_for("full")
    assert metrics.refusal_correctness >= 0.8
    assert metrics.citation_coverage >= 0.95
