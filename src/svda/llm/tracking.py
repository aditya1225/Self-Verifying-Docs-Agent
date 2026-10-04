"""Usage-accumulating decorator around any LLM client.

Cost per query is one of the headline ablation columns, so token usage has to
be collected at the one place every call passes through rather than threaded
back through each stage's return type.
"""

from __future__ import annotations

from ..schemas import Usage
from .base import LLMClient, LLMRequest, LLMResponse, Task


class UsageTrackingLLM:
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.name = f"{inner.name}"
        self.total = Usage()
        self._pending = Usage()
        self.per_task: dict[str, Usage] = {}

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await self._inner.complete(request)
        self.total = self.total.merge(response.usage)
        self._pending = self._pending.merge(response.usage)
        key = request.task.value if isinstance(request.task, Task) else str(request.task)
        self.per_task[key] = self.per_task.get(key, Usage()).merge(response.usage)
        return response

    def take_usage(self) -> Usage:
        """Usage since the last call. Lets the pipeline attribute spend per stage."""
        pending, self._pending = self._pending, Usage()
        return pending

    def reset(self) -> None:
        self.total = Usage()
        self._pending = Usage()
        self.per_task.clear()

    def __getattr__(self, item):  # pragma: no cover - transparent delegation
        return getattr(self._inner, item)
