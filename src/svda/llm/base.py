"""LLM abstraction shared by the router, planner, generator and critic.

Every call carries a `task` tag. The real client renders `prompt`; the offline
stub dispatches on `task` and reads the structured `payload`, so the same
pipeline runs end-to-end with or without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..schemas import Usage


class Task(str, Enum):
    ROUTE = "route"
    PLAN = "plan"
    GENERATE = "generate"
    CRITIQUE = "critique"
    INJECTION_SCAN = "injection_scan"


class Tier(str, Enum):
    CHEAP = "cheap"
    STRONG = "strong"


@dataclass
class LLMRequest:
    task: Task
    prompt: str
    system: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    tier: Tier = Tier.CHEAP
    max_tokens: int = 4096
    effort: str | None = None


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Usage
    truncated: bool = False
    refused: bool = False


class LLMClient(Protocol):
    """Minimal surface the agent depends on."""

    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - protocol
        ...


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    from ..config import get_settings

    in_price, out_price = get_settings().price_for(model)
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
