"""Human-in-the-loop review queue (Ch. 13, stretch goal).

Low-confidence answers are appended to a JSONL queue instead of being shipped.
A reviewer marks each one accepted or corrected, and `svda review promote`
turns the reviewed items into new golden-set entries - closing the loop
between production failures and the eval set.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..schemas import AgentAnswer, GoldenItem


@dataclass
class ReviewItem:
    id: str
    created_at: str
    query: str
    answer: str
    status: str  # pipeline status at the time of queueing
    reason: str
    faithfulness: float
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str = ""
    reviewed: bool = False
    reviewer_verdict: str = ""  # "accept" | "reject" | "corrected"
    corrected_answer: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class ReviewQueue:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().review_queue_path

    def enqueue(self, answer: AgentAnswer, reason: str) -> ReviewItem:
        item = ReviewItem(
            id=uuid.uuid4().hex[:12],
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            query=answer.query,
            answer=answer.answer,
            status=answer.status.value,
            reason=reason,
            faithfulness=answer.critic.faithfulness if answer.critic else 0.0,
            citations=[c.model_dump() for c in answer.citations],
            trace_id=answer.trace_id,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(item.to_json() + "\n")
        return item

    def load(self) -> list[ReviewItem]:
        if not self.path.exists():
            return []
        items: list[ReviewItem] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    items.append(ReviewItem(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return items

    def pending(self) -> list[ReviewItem]:
        return [i for i in self.load() if not i.reviewed]

    def record_verdict(self, item_id: str, verdict: str, corrected_answer: str = "") -> bool:
        items = self.load()
        found = False
        for item in items:
            if item.id == item_id:
                item.reviewed = True
                item.reviewer_verdict = verdict
                item.corrected_answer = corrected_answer
                found = True
        if found:
            with self.path.open("w", encoding="utf-8") as fh:
                for item in items:
                    fh.write(item.to_json() + "\n")
        return found

    def to_golden_items(self) -> list[GoldenItem]:
        """Reviewed + corrected items become regression cases."""
        from ..llm.stub import content_terms

        out: list[GoldenItem] = []
        for item in self.load():
            if not item.reviewed or item.reviewer_verdict == "reject":
                continue
            answer = item.corrected_answer or item.answer
            keywords = sorted(content_terms(answer))[:6]
            out.append(
                GoldenItem(
                    id=f"review-{item.id}",
                    question=item.query,
                    answer_keywords=keywords,
                    context_keywords=keywords[:4],
                    notes=f"promoted from review queue on {item.created_at}",
                )
            )
        return out
