"""Guardrails (Ch. 18) on all three untrusted surfaces.

1. Input   - the user's query may try to redirect the agent.
2. Document- an ingested PDF may carry an injected instruction; caught at
             ingest time so a poisoned chunk never enters the index.
3. Output  - every sentence must carry a resolvable citation, and the answer
             must not leak the prompt or invent a source.

These are deterministic pattern checks, which is deliberate: a guardrail that
needs an LLM call fails open exactly when the LLM is the thing under attack.
An optional LLM second opinion is layered on top for the input surface only.
"""

from __future__ import annotations

import re

from ..schemas import (
    Citation,
    Draft,
    GuardrailStage,
    GuardrailVerdict,
    GuardrailViolation,
)

MAX_QUERY_CHARS = 2000

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|preceding)\s+(instructions?|prompts?|rules?)",
     "instruction override"),
    (r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above)", "instruction override"),
    (r"forget\s+(everything|all)\s+(you|above|previously)", "instruction override"),
    (r"you\s+are\s+now\s+(a|an|the)\b", "persona hijack"),
    (r"\b(reveal|print|repeat|output|show)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules)",
     "prompt extraction"),
    (r"</?\s*(system|instructions?|admin)\s*>", "fake system delimiter"),
    (r"\[\s*(system|admin)\s*\]", "fake system delimiter"),
    (r"\bdeveloper\s+mode\b", "jailbreak marker"),
    (r"\bDAN\b", "jailbreak marker"),
    (r"without\s+(citing|citations?|sources?)", "citation-suppression attempt"),
    (r"\b(do\s+not|don't|never)\s+(cite|mention)\s+(the\s+)?(sources?|citations?|documents?)",
     "citation-suppression attempt"),
    (r"exfiltrat", "exfiltration marker"),
    (r"\bcurl\s+https?://", "outbound request injection"),
    (r"base64\s+-d", "obfuscated payload"),
    (r"send\s+(this|the|all)\s+.{0,30}\bto\s+https?://", "exfiltration marker"),
]

# Patterns that only matter inside ingested documents.
_DOCUMENT_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)\bAI\s+(assistant|agent|model)[^.]{0,40}\b(must|should|shall)\s+(ignore|disregard|always)",
     "document addresses the model directly"),
    (r"(?i)when\s+(asked|answering)[^.]{0,60}\balways\s+(say|answer|respond|reply)",
     "document dictates answers"),
]

_PROMPT_LEAK_PATTERNS = [
    r"(?i)you are a (careful|helpful) (research|retrieval) assistant",
    r"(?i)^\s*system\s*:",
    r"(?i)here (is|are) my (system )?(prompt|instructions)",
]

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def _scan(text: str, patterns: list[tuple[str, str]]) -> list[GuardrailViolation]:
    found: list[GuardrailViolation] = []
    seen: set[str] = set()
    for pattern, label in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and label not in seen:
            seen.add(label)
            found.append(
                GuardrailViolation(
                    rule="prompt_injection",
                    severity="block",
                    detail=f"{label}: matched {match.group(0)[:60]!r}",
                )
            )
    return found


def scan_input(query: str) -> GuardrailVerdict:
    violations: list[GuardrailViolation] = []

    if not query or not query.strip():
        violations.append(GuardrailViolation(rule="empty_query", detail="query is empty"))
    elif len(query) > MAX_QUERY_CHARS:
        violations.append(
            GuardrailViolation(
                rule="query_too_long",
                detail=f"{len(query)} chars exceeds the {MAX_QUERY_CHARS} limit",
            )
        )

    violations.extend(_scan(query or "", _INJECTION_PATTERNS))

    return GuardrailVerdict(
        stage=GuardrailStage.INPUT,
        passed=not any(v.severity == "block" for v in violations),
        violations=violations,
    )


def scan_document(text: str) -> GuardrailVerdict:
    violations = _scan(text, _INJECTION_PATTERNS) + _scan(text, _DOCUMENT_PATTERNS)
    return GuardrailVerdict(
        stage=GuardrailStage.DOCUMENT,
        passed=not any(v.severity == "block" for v in violations),
        violations=violations,
    )


def check_output(
    draft: Draft,
    citations: list[Citation],
    *,
    require_citation_per_sentence: bool = True,
) -> GuardrailVerdict:
    """The video's rule, enforced mechanically: no sentence without a source."""
    violations: list[GuardrailViolation] = []
    valid_markers = {c.marker for c in citations}
    valid_chunk_ids = {c.chunk_id for c in citations}

    for pattern in _PROMPT_LEAK_PATTERNS:
        if re.search(pattern, draft.answer):
            violations.append(
                GuardrailViolation(rule="prompt_leak", detail=f"answer matched {pattern!r}")
            )
            break

    if require_citation_per_sentence:
        for sentence in draft.sentences:
            if not _needs_citation(sentence.text):
                continue
            if not sentence.has_citation:
                violations.append(
                    GuardrailViolation(
                        rule="uncited_sentence",
                        detail=f"no citation on: {sentence.text[:80]!r}",
                    )
                )
            else:
                unknown = [cid for cid in sentence.citation_ids if cid not in valid_chunk_ids]
                if unknown:
                    violations.append(
                        GuardrailViolation(
                            rule="dangling_citation",
                            detail=f"sentence cites unknown chunk(s) {unknown}",
                        )
                    )

    # A marker in the prose that has no matching citation entry is a fabricated source.
    for marker_number in set(_CITATION_MARKER_RE.findall(draft.answer)):
        if f"[{marker_number}]" not in valid_markers:
            violations.append(
                GuardrailViolation(
                    rule="fabricated_citation_marker",
                    detail=f"answer references [{marker_number}] which is not in the citation list",
                )
            )

    return GuardrailVerdict(
        stage=GuardrailStage.OUTPUT,
        passed=not any(v.severity == "block" for v in violations),
        violations=violations,
    )


_NO_CITATION_NEEDED = re.compile(
    r"^\s*(in short|in summary|to summarise|to summarize|put differently|that is|however|"
    r"i can't|i cannot|i don't|the (book|corpus|documents?) (does not|doesn't))",
    re.IGNORECASE,
)


def _needs_citation(sentence: str) -> bool:
    """Transitions and refusals carry no factual claim, so they need no source."""
    stripped = sentence.strip()
    if len(stripped) < 25:
        return False
    if _NO_CITATION_NEEDED.match(stripped):
        return False
    return True
