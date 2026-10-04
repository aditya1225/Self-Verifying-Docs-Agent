"""Tracing: every stage emits a span, every span lands in a JSONL file.

Langfuse is optional. If the SDK and keys are present we mirror spans to it;
otherwise the local JSONL sink is the source of truth and the dashboard reads
from that, so traces are never lost to a missing dependency.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ms: float
    end_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or _now_ms()) - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": _jsonable(self.attributes),
        }


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Tracer:
    """Collects spans for one request and flushes them to disk."""

    def __init__(self, trace_id: str | None = None, sink: Path | None = None) -> None:
        settings = get_settings()
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self._stack: list[str] = []
        self._lock = threading.Lock()
        self.sink = sink if sink is not None else settings.runs_dir / "traces" / "spans.jsonl"
        self._langfuse = _try_langfuse()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(
            name=name,
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:12],
            parent_id=self._stack[-1] if self._stack else None,
            start_ms=_now_ms(),
            attributes=dict(attributes),
        )
        # Registered on entry, not on exit, so a span that is still open - the
        # enclosing `pipeline` span, notably - is visible to `stage_latencies`
        # and `flush` when the request finishes inside it.
        with self._lock:
            self.spans.append(span)
            self._stack.append(span.span_id)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            span.status = "error"
            span.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_ms = _now_ms()
            with self._lock:
                if self._stack and self._stack[-1] == span.span_id:
                    self._stack.pop()

    def event(self, name: str, **attributes: Any) -> None:
        """A zero-duration span, for decisions worth seeing in the trace."""
        span = Span(
            name=name,
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:12],
            parent_id=self._stack[-1] if self._stack else None,
            start_ms=_now_ms(),
            attributes=dict(attributes),
        )
        span.end_ms = span.start_ms
        with self._lock:
            self.spans.append(span)

    def stage_latencies(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for span in self.spans:
            if span.duration_ms > 0:
                out[span.name] = round(out.get(span.name, 0.0) + span.duration_ms, 2)
        return out

    def flush(self) -> None:
        if not self.spans:
            return
        try:
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a", encoding="utf-8") as fh:
                for span in self.spans:
                    fh.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass  # tracing must never break the request path
        self._export_langfuse()

    def _export_langfuse(self) -> None:
        if self._langfuse is None:
            return
        try:
            for span in self.spans:
                self._langfuse.create_event(
                    name=span.name,
                    trace_context={"trace_id": self.trace_id},
                    metadata=span.to_dict(),
                )
            self._langfuse.flush()
        except Exception:  # noqa: BLE001 - observability is best-effort
            pass


def _try_langfuse():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        return Langfuse()
    except Exception:  # noqa: BLE001
        return None


def load_spans(path: Path | None = None) -> list[dict[str, Any]]:
    """Read spans back out of the JSONL sink (used by the dashboard)."""
    settings = get_settings()
    path = path or settings.runs_dir / "traces" / "spans.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
