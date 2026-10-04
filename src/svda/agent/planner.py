"""Query planning (Ch. 6).

Only multi-hop queries reach the planner. It produces self-contained
sub-questions which the retrieval layer then runs concurrently.
"""

from __future__ import annotations

from ..config import get_settings
from ..llm import LLMClient, LLMRequest, Task, Tier
from ..reliability import Degradation, parse_json_lenient
from ..schemas import Plan
from ..tracing import Tracer
from . import prompts


class Planner:
    def __init__(self, llm: LLMClient, tracer: Tracer) -> None:
        self.llm = llm
        self.tracer = tracer
        self.settings = get_settings()

    async def plan(
        self, query: str, *, degradations: list[Degradation] | None = None
    ) -> Plan:
        with self.tracer.span("planner", query=query) as span:
            request = LLMRequest(
                task=Task.PLAN,
                prompt=prompts.plan_prompt(query, self.settings.max_sub_questions),
                system=prompts.SYSTEM_BASE,
                payload={"query": query, "max_sub_questions": self.settings.max_sub_questions},
                tier=Tier.CHEAP,
                max_tokens=512,
            )
            try:
                response = await self.llm.complete(request)
                data = parse_json_lenient(response.text)
                subs = [
                    str(s).strip()
                    for s in data.get("sub_questions", [])
                    if str(s).strip()
                ][: self.settings.max_sub_questions]
                if not subs:
                    subs = [query]
                plan = Plan(
                    original_query=query,
                    sub_questions=subs,
                    rationale=str(data.get("rationale", ""))[:200],
                    planned_by="llm",
                )
            except Exception as exc:  # noqa: BLE001
                if degradations is not None:
                    degradations.append(Degradation("planner", f"{type(exc).__name__}: {exc}"))
                plan = Plan(
                    original_query=query,
                    sub_questions=[query],
                    rationale=f"planner unavailable ({type(exc).__name__}); querying as-is",
                    planned_by="fallback",
                )

            span.attributes["sub_questions"] = plan.sub_questions
            span.attributes["planned_by"] = plan.planned_by
            return plan

    @staticmethod
    def passthrough(query: str) -> Plan:
        """The ablation baseline: no decomposition at all."""
        return Plan(original_query=query, sub_questions=[query], rationale="planning disabled")
