"""LLM backends and the factory that picks one."""

from __future__ import annotations

from ..config import get_settings
from ..reliability import Degradation
from .base import LLMClient, LLMRequest, LLMResponse, Task, Tier, estimate_cost
from .stub import StubLLM
from .tracking import UsageTrackingLLM

__all__ = [
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "Task",
    "Tier",
    "StubLLM",
    "UsageTrackingLLM",
    "build_llm",
    "backend_name",
    "estimate_cost",
]


def _build_backend(degradations: list[Degradation] | None = None) -> LLMClient:
    settings = get_settings()
    if settings.force_stub_llm or not settings.anthropic_api_key:
        return StubLLM()
    try:
        from .anthropic_client import AnthropicLLM

        return AnthropicLLM(degradations=degradations)
    except Exception as exc:  # noqa: BLE001 - never block on a backend import
        if degradations is not None:
            degradations.append(Degradation("llm", f"anthropic client unavailable: {exc}"))
        return StubLLM()


def build_llm(degradations: list[Degradation] | None = None) -> UsageTrackingLLM:
    """Real client when a key is configured, deterministic stub otherwise.

    Falling back to the stub rather than raising is deliberate: `svda verify`,
    the test suite and CI all run on machines with no credentials.
    """
    return UsageTrackingLLM(_build_backend(degradations))


def backend_name() -> str:
    """Which backend `build_llm` would pick, without constructing a client."""
    settings = get_settings()
    if settings.force_stub_llm or not settings.anthropic_api_key:
        return "stub"
    return f"anthropic:{settings.model_strong}/{settings.model_cheap}"
