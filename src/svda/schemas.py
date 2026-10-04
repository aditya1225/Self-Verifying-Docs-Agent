"""Typed contracts between pipeline steps (Phase 1: every step has a typed in/out).

Because each stage consumes and produces one of these models, a stage can be
swapped (or ablated away) without touching its neighbours.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


class Chunk(BaseModel):
    """One retrievable unit of the corpus, carrying its own citation metadata."""

    id: str
    doc_id: str
    text: str
    chapter: str = "Unknown"
    chapter_number: int | None = None
    section: str = "Unknown"
    page_start: int = 0
    page_end: int = 0
    ordinal: int = 0

    @property
    def citation_label(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.chapter} > {self.section}, {pages}"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str
    rank: int = 0


# --------------------------------------------------------------------------
# Guardrails (Ch. 18)
# --------------------------------------------------------------------------


class GuardrailStage(str, Enum):
    INPUT = "input"
    DOCUMENT = "document"
    OUTPUT = "output"


class GuardrailViolation(BaseModel):
    rule: str
    severity: str = "block"  # "block" | "warn"
    detail: str = ""


class GuardrailVerdict(BaseModel):
    stage: GuardrailStage
    passed: bool
    violations: list[GuardrailViolation] = Field(default_factory=list)

    @property
    def blocking(self) -> list[GuardrailViolation]:
        return [v for v in self.violations if v.severity == "block"]


# --------------------------------------------------------------------------
# Routing (Ch. 2)
# --------------------------------------------------------------------------


class Route(str, Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    OUT_OF_SCOPE = "out_of_scope"
    CHITCHAT = "chitchat"


class RouteDecision(BaseModel):
    route: Route
    confidence: float = 0.5
    reason: str = ""
    model_tier: str = "cheap"  # which tier the generator should use (Ch. 16)
    decided_by: str = "heuristic"  # "heuristic" | "llm" | "fallback"


# --------------------------------------------------------------------------
# Planning (Ch. 6)
# --------------------------------------------------------------------------


class Plan(BaseModel):
    original_query: str
    sub_questions: list[str] = Field(default_factory=list)
    rationale: str = ""
    planned_by: str = "none"  # "none" | "llm" | "fallback"

    @property
    def is_decomposed(self) -> bool:
        return len(self.sub_questions) > 1


# --------------------------------------------------------------------------
# Retrieval (Ch. 3)
# --------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    query: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    per_retriever_counts: dict[str, int] = Field(default_factory=dict)
    reranked: bool = False
    degraded: bool = False  # a retriever or the re-ranker failed and we fell back
    notes: list[str] = Field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [sc.chunk.id for sc in self.chunks]


# --------------------------------------------------------------------------
# Generation (Ch. 14)
# --------------------------------------------------------------------------


class Sentence(BaseModel):
    """One sentence of the draft plus the citations attached to it."""

    text: str
    citation_ids: list[str] = Field(default_factory=list)

    @property
    def has_citation(self) -> bool:
        return bool(self.citation_ids)


class Draft(BaseModel):
    answer: str
    sentences: list[Sentence] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    model: str = "stub"
    attempt: int = 1
    declined: bool = False


# --------------------------------------------------------------------------
# Reflection (Ch. 4)
# --------------------------------------------------------------------------


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    evidence_chunk_id: str | None = None
    reason: str = ""


class CriticAction(str, Enum):
    ACCEPT = "accept"
    REGENERATE = "regenerate"
    DECLINE = "decline"
    HUMAN_REVIEW = "human_review"


class CriticVerdict(BaseModel):
    faithfulness: float = 0.0  # fraction of checked claims that are supported
    citation_coverage: float = 0.0  # fraction of sentences carrying a citation
    claims: list[ClaimCheck] = Field(default_factory=list)
    action: CriticAction = CriticAction.ACCEPT
    reason: str = ""
    checked_by: str = "heuristic"  # "heuristic" | "llm" | "fallback"


# --------------------------------------------------------------------------
# Final answer
# --------------------------------------------------------------------------


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    DECLINED = "declined"
    BLOCKED = "blocked"
    HUMAN_REVIEW = "human_review"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            llm_calls=self.llm_calls + other.llm_calls,
        )


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    chunk_id: str
    label: str
    chapter: str
    section: str
    page_start: int
    page_end: int
    quote: str = ""


class AgentAnswer(BaseModel):
    """The one object the API, CLI, MCP server and evaluator all consume."""

    query: str
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: RouteDecision | None = None
    plan: Plan | None = None
    critic: CriticVerdict | None = None
    input_guardrail: GuardrailVerdict | None = None
    output_guardrail: GuardrailVerdict | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_degraded: bool = False
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str = ""
    profile: str = "full"
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.status in {AnswerStatus.DECLINED, AnswerStatus.BLOCKED}


# --------------------------------------------------------------------------
# Evaluation (Ch. 19)
# --------------------------------------------------------------------------


class GoldenItem(BaseModel):
    id: str
    question: str
    kind: Route = Route.SIMPLE
    answerable: bool = True
    # Terms that a correct answer should contain (lowercased substring match).
    answer_keywords: list[str] = Field(default_factory=list)
    # Terms that correctly retrieved context should contain. Used for
    # retrieval precision/recall without pinning to chunk ids, so the metric
    # survives a re-chunk.
    context_keywords: list[str] = Field(default_factory=list)
    expected_chapters: list[int] = Field(default_factory=list)
    notes: str = ""


class ItemResult(BaseModel):
    item_id: str
    question: str
    kind: Route
    answerable: bool
    status: AnswerStatus
    answer: str
    correct: bool = False
    faithful: bool = False
    faithfulness: float = 0.0
    citation_coverage: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_precision: float = 0.0
    chapter_hit: bool = False
    refused_correctly: bool = False
    route_correct: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class ProfileMetrics(BaseModel):
    profile: str
    n: int = 0
    answer_accuracy: float = 0.0
    faithfulness: float = 0.0
    citation_coverage: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_precision: float = 0.0
    multihop_accuracy: float = 0.0
    refusal_correctness: float = 0.0
    route_accuracy: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    errors: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    run_id: str
    created_at: str
    llm_backend: str
    profiles: list[ProfileMetrics] = Field(default_factory=list)
    items: dict[str, list[ItemResult]] = Field(default_factory=dict)

    def metrics_for(self, profile: str) -> ProfileMetrics | None:
        return next((p for p in self.profiles if p.profile == profile), None)
