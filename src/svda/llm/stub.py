"""Deterministic offline LLM.

This is what makes the repo self-verifying: the full agent graph - routing,
planning, grounded generation with citations, and the faithfulness critic -
runs with no network and no API key, producing the same output every time.

The stub is genuinely *extractive*: it answers only with sentences lifted from
the retrieved chunks, and its critic re-checks those sentences against the
chunks. So an ungrounded answer is still caught offline, and a test that
asserts "the agent declines when the corpus lacks the answer" means something.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schemas import Usage
from .base import LLMRequest, LLMResponse, Task

_STOPWORDS = {
    "a", "about", "above", "after", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "between", "both", "but", "by", "can", "did", "do", "does",
    "doing", "during", "each", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "interact", "into", "is", "it", "its", "me", "more", "most", "my",
    "no", "nor", "not", "of", "on", "once", "only", "or", "other", "our", "out",
    "over", "own", "same", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with", "would",
    "you", "your", "explain", "describe", "tell", "give",
}

_MULTIHOP_MARKERS = (
    " and how ", " and why ", " interact", " compare", " versus ", " vs ",
    " difference between ", " trade-off", " tradeoff", " relate", " affect",
    " implications of", " combined with", " together with",
)

_CHITCHAT_PATTERNS = (
    r"^\s*(hi|hey|hello|yo|sup)\b",
    r"^\s*(thanks|thank you|ty|cheers)\b",
    r"^\s*(good (morning|afternoon|evening))\b",
    r"^\s*how are you\b",
    r"^\s*(bye|goodbye|see ya)\b",
    r"^\s*who are you\b",
    r"^\s*what can you do\b",
)

_INJECTION_PATTERNS = (
    r"ignore (all |any |the )?(previous|prior|above) instructions",
    r"disregard (all |any |the )?(previous|prior|above)",
    r"you are now\b",
    r"system prompt",
    r"reveal your (instructions|prompt|system)",
    r"print your (instructions|prompt|rules)",
    r"</?(system|instructions?)>",
    r"\bDAN\b",
    r"developer mode",
    r"exfiltrat",
    r"curl\s+http",
    r"base64 -d",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-']+", text.lower()) if t not in _STOPWORDS]


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 2}


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class StubLLM:
    """Offline, deterministic, zero-cost. Reads `request.payload`, not the prompt."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[Task] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request.task)
        handler = {
            Task.ROUTE: self._route,
            Task.PLAN: self._plan,
            Task.GENERATE: self._generate,
            Task.CRITIQUE: self._critique,
            Task.INJECTION_SCAN: self._injection_scan,
        }[request.task]
        body = handler(request.payload)
        text = json.dumps(body, ensure_ascii=False)
        # Token counts are character-based estimates so cost accounting and
        # dashboards have non-zero, reproducible numbers offline.
        usage = Usage(
            input_tokens=len(request.prompt) // 4,
            output_tokens=len(text) // 4,
            cost_usd=0.0,
            llm_calls=1,
        )
        return LLMResponse(text=text, model="stub", usage=usage)

    # -- route ------------------------------------------------------------
    def _route(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = (payload.get("query") or "").strip()
        lowered = f" {query.lower()} "

        for pattern in _CHITCHAT_PATTERNS:
            if re.search(pattern, query.lower()):
                return {"route": "chitchat", "confidence": 0.9, "reason": "greeting or small talk"}

        corpus_terms: set[str] = {t.lower() for t in payload.get("corpus_vocabulary", [])}
        terms = content_terms(query)
        overlap = len(terms & corpus_terms) / max(len(terms), 1) if corpus_terms else 1.0

        if corpus_terms and overlap < 0.2 and len(terms) >= 2:
            return {
                "route": "out_of_scope",
                "confidence": 0.7,
                "reason": f"only {overlap:.0%} of query terms appear in the corpus vocabulary",
            }

        hops = sum(1 for marker in _MULTIHOP_MARKERS if marker in lowered)
        question_marks = query.count("?")
        if hops >= 1 or question_marks > 1:
            return {
                "route": "multi_hop",
                "confidence": 0.75,
                "reason": "question couples two concepts or asks several things at once",
            }
        return {"route": "simple", "confidence": 0.8, "reason": "single-concept lookup"}

    # -- plan -------------------------------------------------------------
    def _plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = (payload.get("query") or "").strip()
        limit = int(payload.get("max_sub_questions", 4))

        # Split on explicit multi-question punctuation first.
        explicit = [q.strip() for q in re.split(r"\?\s+", query) if q.strip()]
        if len(explicit) > 1:
            subs = [q if q.endswith("?") else q + "?" for q in explicit]
            return {"sub_questions": subs[:limit], "rationale": "query contained several questions"}

        # Otherwise split around the coupling marker: each side becomes a hop,
        # plus one question about the interaction itself.
        lowered = query.lower()
        for marker in (" interact", " and how ", " and why ", " versus ", " vs ",
                       " difference between ", " compare", " relate", " affect"):
            idx = lowered.find(marker)
            if idx == -1:
                continue
            left = query[:idx].strip(" ,?")
            right = query[idx + len(marker):].strip(" ,?")
            subs = []
            if left:
                subs.append(f"What is {_strip_lead(left)}?")
            if right:
                subs.append(f"What is {_strip_lead(right)}?")
            subs.append(query if query.endswith("?") else query + "?")
            return {
                "sub_questions": subs[:limit],
                "rationale": f"decomposed around '{marker.strip()}' into per-concept hops",
            }

        return {"sub_questions": [query], "rationale": "no decomposition needed"}

    # -- generate ---------------------------------------------------------
    def _generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query") or ""
        contexts: list[dict[str, Any]] = payload.get("contexts", [])
        if not contexts:
            return {"declined": True, "sentences": [], "reason": "no context retrieved"}

        query_terms = content_terms(query)
        scored: list[tuple[float, int, str]] = []
        for idx, ctx in enumerate(contexts, start=1):
            for sentence in split_sentences(ctx.get("text", "")):
                if len(sentence) < 40 or len(sentence) > 400:
                    continue
                sentence_terms = content_terms(sentence)
                if not sentence_terms:
                    continue
                overlap = len(query_terms & sentence_terms)
                if overlap == 0:
                    continue
                # Favour sentences dense in query terms, and earlier contexts
                # (the re-ranker already ordered them).
                score = overlap / (len(query_terms) ** 0.5) + 0.35 * overlap / len(sentence_terms)
                score -= 0.03 * (idx - 1)
                scored.append((score, idx, sentence.strip()))

        if not scored:
            return {"declined": True, "sentences": [], "reason": "no retrieved sentence matches the question"}

        scored.sort(key=lambda row: (-row[0], row[1]))
        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _, idx, sentence in scored:
            key = sentence[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append((idx, sentence))
            if len(chosen) >= 4:
                break

        return {
            "declined": False,
            "sentences": [{"text": sentence, "citations": [idx]} for idx, sentence in chosen],
        }

    # -- critique ---------------------------------------------------------
    def _critique(self, payload: dict[str, Any]) -> dict[str, Any]:
        contexts: list[dict[str, Any]] = payload.get("contexts", [])
        sentences: list[dict[str, Any]] = payload.get("sentences", [])
        by_index = {idx: (ctx.get("text") or "").lower() for idx, ctx in enumerate(contexts, start=1)}
        all_text = " ".join(by_index.values())

        claims = []
        for sentence in sentences:
            text = sentence.get("text", "")
            cites = [int(c) for c in sentence.get("citations", []) if str(c).isdigit()]
            terms = content_terms(text)
            if not terms:
                continue

            cited_text = " ".join(by_index.get(c, "") for c in cites)
            haystack = cited_text or all_text
            covered = sum(1 for term in terms if term in haystack)
            ratio = covered / len(terms)

            supported = bool(cites) and ratio >= 0.85
            claims.append(
                {
                    "claim": text,
                    "supported": supported,
                    "evidence": cites[0] if cites else None,
                    "reason": (
                        "every content term appears in the cited passage"
                        if supported
                        else (
                            "no citation attached"
                            if not cites
                            else f"only {ratio:.0%} of content terms appear in the cited passage"
                        )
                    ),
                }
            )

        if not claims:
            return {"claims": [], "verdict": "decline", "reason": "nothing to verify"}

        supported_fraction = sum(1 for c in claims if c["supported"]) / len(claims)
        if supported_fraction >= 0.8:
            verdict = "accept"
        elif supported_fraction >= 0.4:
            verdict = "regenerate"
        else:
            verdict = "decline"
        return {
            "claims": claims,
            "verdict": verdict,
            "reason": f"{supported_fraction:.0%} of claims are supported by the cited passages",
        }

    # -- injection scan ---------------------------------------------------
    def _injection_scan(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = (payload.get("text") or "").lower()
        hits = [p for p in _INJECTION_PATTERNS if re.search(p, text)]
        return {
            "suspicious": bool(hits),
            "reason": f"matched {len(hits)} injection pattern(s)" if hits else "no injection markers",
            "patterns": hits[:5],
        }


def _strip_lead(text: str) -> str:
    """Turn a clause into a noun phrase for a sub-question."""
    cleaned = re.sub(
        r"^(how does|how do|how is|how are|what is|what are|why does|why do|does|do|is|are|the)\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip() or text.strip()
