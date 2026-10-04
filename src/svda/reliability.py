"""Exception handling (Ch. 12): timeouts, retries, JSON repair, fallbacks.

The rule this module enforces: a degraded dependency produces a degraded answer,
never an exception that escapes to the caller. Each helper records *why* it
degraded so the trace and the final answer can report it honestly.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class SVDAError(Exception):
    """Base class for errors this package raises deliberately."""


class CorpusNotIndexedError(SVDAError):
    pass


class LLMUnavailableError(SVDAError):
    pass


@dataclass
class Degradation:
    """A recorded fallback, surfaced in the trace and the answer's warnings."""

    component: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.component}: {self.reason}"


async def with_timeout(
    coro: Awaitable[T],
    timeout_s: float,
    *,
    component: str,
    fallback: T,
    degradations: list[Degradation] | None = None,
) -> T:
    """Await `coro`, returning `fallback` if it times out or raises."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        if degradations is not None:
            degradations.append(Degradation(component, f"timed out after {timeout_s}s"))
        return fallback
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the boundary
        if degradations is not None:
            degradations.append(Degradation(component, f"{type(exc).__name__}: {exc}"))
        return fallback


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    component: str = "unknown",
    degradations: list[Degradation] | None = None,
) -> T:
    """Exponential backoff. Re-raises the last error if every attempt fails."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except retry_on as exc:
            last = exc
            if attempt == attempts - 1:
                break
            if degradations is not None:
                degradations.append(
                    Degradation(component, f"attempt {attempt + 1} failed: {type(exc).__name__}")
                )
            await asyncio.sleep(base_delay * (2**attempt))
    assert last is not None
    raise last


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_lenient(text: str) -> Any:
    """Parse JSON out of an LLM response that may be wrapped in prose or fences.

    Raises ValueError if nothing parseable is found, so the caller can retry
    with a stricter prompt before falling back.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # One cheap repair pass: drop trailing commas.
            repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no JSON object found in response: {text[:200]!r}")
