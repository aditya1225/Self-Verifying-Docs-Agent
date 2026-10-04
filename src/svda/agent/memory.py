"""Conversational memory (Ch. 8, stretch goal).

Scope is deliberately narrow: carry enough of the last few turns to resolve a
follow-up like "and how does that interact with sharding?" into a
self-contained query. Memory never becomes evidence - only retrieved chunks
can support a claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import get_settings

_FOLLOW_UP_RE = re.compile(
    r"^\s*(and|but|what about|how about|why|then|also|ok|okay)\b|"
    r"\b(that|those|them|it|this|the same|the former|the latter)\b",
    re.IGNORECASE,
)


@dataclass
class Turn:
    query: str
    answer: str
    topic_terms: list[str] = field(default_factory=list)


@dataclass
class ConversationMemory:
    turns: list[Turn] = field(default_factory=list)

    def add(self, query: str, answer: str) -> None:
        self.turns.append(Turn(query=query, answer=answer, topic_terms=_topic_terms(query)))
        limit = get_settings().memory_turns
        if len(self.turns) > limit:
            self.turns = self.turns[-limit:]

    def contextualize(self, query: str) -> tuple[str, bool]:
        """Return (query_for_retrieval, was_rewritten).

        The rewrite only ever *adds* prior topic terms - it cannot change what
        the user asked, which keeps the failure mode to "retrieved too much"
        rather than "answered a different question".
        """
        if not self.turns or not _looks_like_follow_up(query):
            return query, False

        prior: list[str] = []
        for turn in reversed(self.turns[-2:]):
            for term in turn.topic_terms:
                if term not in prior and term.lower() not in query.lower():
                    prior.append(term)
        if not prior:
            return query, False
        return f"{query} (in the context of: {', '.join(prior[:6])})", True

    def clear(self) -> None:
        self.turns.clear()


def _looks_like_follow_up(query: str) -> bool:
    words = query.split()
    return bool(_FOLLOW_UP_RE.search(query)) and len(words) <= 20


_TOPIC_STOPWORDS = {
    "what", "how", "why", "does", "do", "is", "are", "the", "a", "an", "of",
    "in", "to", "and", "for", "with", "that", "this", "it", "explain", "describe",
}


def _topic_terms(query: str) -> list[str]:
    terms = [
        w.strip(".,?!'\"")
        for w in query.split()
        if w.strip(".,?!'\"").lower() not in _TOPIC_STOPWORDS and len(w) > 3
    ]
    return terms[:6]
