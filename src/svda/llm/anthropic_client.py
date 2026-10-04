"""Anthropic-backed LLM client with tiering, timeouts and a cheap-model fallback."""

from __future__ import annotations

import asyncio

from ..config import get_settings
from ..reliability import Degradation, LLMUnavailableError
from ..schemas import Usage
from .base import LLMRequest, LLMResponse, Task, Tier, estimate_cost

# Tasks that benefit from reasoning depth; the rest run at low effort to keep
# cost and latency down (Ch. 16).
_HIGH_EFFORT_TASKS = {Task.GENERATE, Task.CRITIQUE, Task.PLAN}


class AnthropicLLM:
    """Wraps the official SDK. Falls back strong -> cheap on failure."""

    name = "anthropic"

    def __init__(self, degradations: list[Degradation] | None = None) -> None:
        import anthropic

        settings = get_settings()
        if not settings.anthropic_api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=settings.max_retries,
        )
        self._settings = settings
        self._degradations = degradations if degradations is not None else []

    def model_for(self, tier: Tier) -> str:
        return self._settings.model_strong if tier is Tier.STRONG else self._settings.model_cheap

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = self.model_for(request.tier)
        try:
            return await self._call(model, request)
        except self._anthropic.APIStatusError as exc:
            if exc.status_code < 500 and exc.status_code != 429:
                raise
            self._degradations.append(
                Degradation("llm", f"{model} returned {exc.status_code}; retrying on cheap tier")
            )
        except (self._anthropic.APIConnectionError, self._anthropic.APITimeoutError, asyncio.TimeoutError) as exc:
            self._degradations.append(
                Degradation("llm", f"{model} unavailable ({type(exc).__name__}); retrying on cheap tier")
            )

        # Fallback: the smaller, faster model. If that fails too, the caller's
        # timeout wrapper turns it into a graceful decline.
        cheap = self._settings.model_cheap
        if cheap == model:
            raise LLMUnavailableError(f"{model} failed and no cheaper tier is configured")
        return await self._call(cheap, request)

    async def _call(self, model: str, request: LLMRequest) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system

        effort = request.effort or ("high" if request.task in _HIGH_EFFORT_TASKS else "low")
        if _supports_effort(model):
            kwargs["output_config"] = {"effort": effort}
            kwargs["thinking"] = {"type": "adaptive"}

        message = await self._client.messages.create(**kwargs)

        text = "".join(block.text for block in message.content if block.type == "text")
        usage = Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cost_usd=estimate_cost(model, message.usage.input_tokens, message.usage.output_tokens),
            llm_calls=1,
        )
        return LLMResponse(
            text=text,
            model=model,
            usage=usage,
            truncated=message.stop_reason == "max_tokens",
            refused=message.stop_reason == "refusal",
        )


def _supports_effort(model: str) -> bool:
    """`output_config.effort` + adaptive thinking are not available on Haiku 4.5."""
    return not model.startswith("claude-haiku")
