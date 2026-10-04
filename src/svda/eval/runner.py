"""Ablation runner (Ch. 19).

Runs the same golden set through every profile and produces the headline
table: naive -> + hybrid/re-rank -> + routing/planning -> + reflection,
measured on faithfulness, multi-hop accuracy, cost and latency.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import uuid
from pathlib import Path

from ..agent.critic import Critic
from ..agent.generator import build_contexts
from ..agent.pipeline import PROFILES, Agent, Profile
from ..config import get_settings
from ..index.store import CorpusIndex
from ..llm import backend_name, build_llm
from ..schemas import (
    AgentAnswer,
    AnswerStatus,
    EvalReport,
    GoldenItem,
    ItemResult,
    RetrievalResult,
    ScoredChunk,
)
from ..tracing import Tracer
from .golden import load_golden
from .metrics import aggregate, extractive_faithfulness, score_item

DEFAULT_PROFILE_ORDER = ["naive", "hybrid", "agentic", "full"]


async def _judge_faithfulness(
    item: GoldenItem,
    answer: AgentAnswer,
    index: CorpusIndex,
    judge: Critic | None,
) -> float:
    """Independent grading - never reads `answer.critic`.

    With a real LLM configured we use a separate critic instance over the
    chunks the pipeline retrieved. Offline we fall back to the deterministic
    extractive check so every profile still gets a comparable number.
    """
    if judge is None or answer.status is not AnswerStatus.ANSWERED or not answer.citations:
        return extractive_faithfulness(answer, index)

    chunks = [index.get(cid) for cid in answer.retrieved_chunk_ids]
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        return extractive_faithfulness(answer, index)

    retrieval = RetrievalResult(
        query=item.question,
        chunks=[
            ScoredChunk(chunk=c, score=1.0, retriever="eval", rank=i)
            for i, c in enumerate(chunks, start=1)
        ],
    )
    contexts = build_contexts(retrieval, get_settings().final_context_chunks)

    from ..schemas import Draft, Sentence
    import re

    sentences = [
        Sentence(
            text=re.sub(r"\[\d+\]", "", s).strip(),
            citation_ids=[
                c.chunk_id for c in answer.citations if c.marker in re.findall(r"\[\d+\]", s)
            ],
        )
        for s in re.split(r"(?<=[.!?])\s+", answer.answer)
        if s.strip()
    ]
    draft = Draft(answer=answer.answer, sentences=sentences)
    verdict = await judge.check(item.question, draft, contexts)
    return verdict.faithfulness


async def run_profile(
    index: CorpusIndex,
    items: list[GoldenItem],
    profile: Profile,
    *,
    concurrency: int = 4,
    use_llm_judge: bool = False,
    progress=None,
) -> list[ItemResult]:
    semaphore = asyncio.Semaphore(concurrency)
    judge = Critic(build_llm(), Tracer()) if use_llm_judge else None

    async def one(item: GoldenItem) -> ItemResult:
        async with semaphore:
            agent = Agent(index, profile=profile)
            try:
                answer = await agent.answer(item.question)
            except Exception as exc:  # noqa: BLE001 - one bad item must not kill the run
                result = ItemResult(
                    item_id=item.id,
                    question=item.question,
                    kind=item.kind,
                    answerable=item.answerable,
                    status=AnswerStatus.DECLINED,
                    answer="",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if progress is not None:
                    progress()
                return result

            faith = await _judge_faithfulness(item, answer, index, judge)
            result = score_item(item, answer, index, judged_faithfulness=faith)
            if progress is not None:
                progress()
            return result

    return list(await asyncio.gather(*(one(item) for item in items)))


async def run_ablation(
    index: CorpusIndex,
    *,
    golden_path: Path | None = None,
    profiles: list[str] | None = None,
    limit: int | None = None,
    concurrency: int = 4,
    use_llm_judge: bool = False,
    progress=None,
) -> EvalReport:
    items = load_golden(golden_path)
    if limit:
        items = items[:limit]

    names = profiles or DEFAULT_PROFILE_ORDER
    report = EvalReport(
        run_id=uuid.uuid4().hex[:10],
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        llm_backend=backend_name(),
    )

    for name in names:
        profile = PROFILES[name]
        results = await run_profile(
            index,
            items,
            profile,
            concurrency=concurrency,
            use_llm_judge=use_llm_judge,
            progress=progress,
        )
        report.items[name] = results
        report.profiles.append(aggregate(name, results))

    return report


def save_report(report: EvalReport, directory: Path | None = None) -> Path:
    directory = directory or get_settings().runs_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"eval-{report.run_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (directory / "latest.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_report(path: Path) -> EvalReport:
    return EvalReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
