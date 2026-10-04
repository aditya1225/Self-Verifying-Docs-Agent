"""Query routing (Ch. 2) + model tiering (Ch. 16).

A cheap deterministic pass handles the obvious cases (greetings, clearly
in-corpus lookups). Only genuinely ambiguous queries pay for an LLM call, and
the route then decides which model tier the generator gets.
"""

from __future__ import annotations

import re

from ..config import get_settings
from ..index.store import CorpusIndex, bm25_tokenize
from ..llm import LLMClient, LLMRequest, Task, Tier
from ..reliability import Degradation, parse_json_lenient
from ..schemas import Route, RouteDecision
from ..tracing import Tracer
from . import prompts

_CHITCHAT_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|cheers|good (morning|afternoon|evening)|"
    r"how are you|who are you|what can you do|bye|goodbye)\b",
    re.IGNORECASE,
)

_MULTIHOP_MARKERS = (
    "interact", "and how", "and why", "compare", "versus", " vs ",
    "difference between", "trade-off", "tradeoff", "relate to", "affect",
    "implications of", "combined with", "as opposed to", "rather than",
)

# Terms that are clearly outside a distributed-systems book.
_OFF_TOPIC_HINTS = (
    "react", "tailwind", "css", "recipe", "weather", "stock price", "movie",
    "football", "horoscope", "travel", "diet", "lyrics",
)


class Router:
    def __init__(self, llm: LLMClient, index: CorpusIndex, tracer: Tracer) -> None:
        self.llm = llm
        self.index = index
        self.tracer = tracer
        self.settings = get_settings()

    async def route(
        self,
        query: str,
        *,
        use_llm: bool = True,
        degradations: list[Degradation] | None = None,
    ) -> RouteDecision:
        with self.tracer.span("router", query=query) as span:
            decision = self._heuristic(query)
            if decision is not None:
                span.attributes["route"] = decision.route.value
                span.attributes["decided_by"] = "heuristic"
                return decision

            if not use_llm:
                decision = self._default(query)
                span.attributes["route"] = decision.route.value
                return decision

            decision = await self._llm_route(query, degradations)
            span.attributes["route"] = decision.route.value
            span.attributes["decided_by"] = decision.decided_by
            span.attributes["model_tier"] = decision.model_tier
            return decision

    # -- fast paths --------------------------------------------------------
    def _heuristic(self, query: str) -> RouteDecision | None:
        """Only returns a decision when it is confident; otherwise defers."""
        text = query.strip()
        if not text:
            return RouteDecision(
                route=Route.CHITCHAT, confidence=1.0, reason="empty query", model_tier="cheap"
            )
        lowered = text.lower()

        if _CHITCHAT_RE.match(lowered) and len(text.split()) <= 8:
            return RouteDecision(
                route=Route.CHITCHAT, confidence=0.95, reason="greeting or small talk",
                model_tier="cheap",
            )

        if any(hint in lowered for hint in _OFF_TOPIC_HINTS):
            return RouteDecision(
                route=Route.OUT_OF_SCOPE, confidence=0.85,
                reason="mentions a topic the corpus does not cover", model_tier="cheap",
            )
        return None

    def _default(self, query: str) -> RouteDecision:
        lowered = f" {query.lower()} "
        multi = any(marker in lowered for marker in _MULTIHOP_MARKERS) or query.count("?") > 1
        route = Route.MULTI_HOP if multi else Route.SIMPLE
        return RouteDecision(
            route=route,
            confidence=0.6,
            reason="keyword heuristic (LLM routing disabled)",
            model_tier=self.tier_for(route).value,
            decided_by="heuristic",
        )

    # -- LLM path ----------------------------------------------------------
    async def _llm_route(
        self, query: str, degradations: list[Degradation] | None
    ) -> RouteDecision:
        request = LLMRequest(
            task=Task.ROUTE,
            prompt=prompts.route_prompt(query),
            system=prompts.SYSTEM_BASE,
            payload={"query": query, "corpus_vocabulary": self._vocabulary_sample(query)},
            tier=Tier.CHEAP,  # routing is always cheap - that is the point
            max_tokens=256,
        )
        try:
            response = await self.llm.complete(request)
            data = parse_json_lenient(response.text)
            route = Route(str(data.get("route", "simple")).strip().lower())
            return RouteDecision(
                route=route,
                confidence=float(data.get("confidence", 0.6)),
                reason=str(data.get("reason", ""))[:200],
                model_tier=self.tier_for(route).value,
                decided_by="llm",
            )
        except Exception as exc:  # noqa: BLE001
            if degradations is not None:
                degradations.append(Degradation("router", f"{type(exc).__name__}: {exc}"))
            fallback = self._default(query)
            fallback.decided_by = "fallback"
            fallback.reason = f"router fell back to heuristics ({type(exc).__name__})"
            return fallback

    def _vocabulary_sample(self, query: str) -> list[str]:
        """Which of the query's own terms exist in the corpus.

        Passing only the intersection keeps the StubLLM's scope check honest
        without serialising a 40k-term vocabulary into the payload.
        """
        terms = set(bm25_tokenize(query))
        return sorted(terms & self.index.vocabulary)

    # -- tiering (Ch. 16) --------------------------------------------------
    @staticmethod
    def tier_for(route: Route) -> Tier:
        """Simple lookups ride the cheap model; reasoning gets the strong one."""
        return Tier.STRONG if route is Route.MULTI_HOP else Tier.CHEAP
