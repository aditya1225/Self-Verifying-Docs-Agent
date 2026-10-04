"""Grounded generation with citations (Ch. 14).

The generator emits sentences plus the context-block numbers each one rests
on. We resolve those numbers back to chunk ids here, so a citation is always
checkable against the index rather than a string the model wrote.
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMClient, LLMRequest, Task, Tier
from ..reliability import Degradation, parse_json_lenient
from ..schemas import Citation, Draft, RetrievalResult, Sentence
from ..tracing import Tracer
from . import prompts

MAX_CONTEXT_CHARS = 2400


def _attach_markers(text: str, markers: str) -> str:
    """Place citation markers *inside* the sentence, before its final stop.

    "...on several machines [1]." rather than "...on several machines. [1]" -
    otherwise a sentence split leaves the marker attached to the next
    sentence, and every per-sentence citation check reads the wrong thing.
    """
    text = text.strip()
    if not markers:
        return text
    if text and text[-1] in ".!?":
        return f"{text[:-1].rstrip()} {markers}{text[-1]}"
    return f"{text} {markers}"


def build_contexts(retrieval: RetrievalResult, limit: int) -> list[dict[str, Any]]:
    """Numbered context blocks, in re-ranked order. Block number == citation marker."""
    contexts: list[dict[str, Any]] = []
    for scored in retrieval.chunks[:limit]:
        chunk = scored.chunk
        text = chunk.text
        if len(text) > MAX_CONTEXT_CHARS:
            text = text[:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0] + " ..."
        contexts.append(
            {
                "chunk_id": chunk.id,
                "label": chunk.citation_label,
                "text": text,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
        )
    return contexts


class Generator:
    def __init__(self, llm: LLMClient, tracer: Tracer) -> None:
        self.llm = llm
        self.tracer = tracer

    async def generate(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        *,
        tier: Tier = Tier.CHEAP,
        attempt: int = 1,
        extra_instruction: str = "",
        degradations: list[Degradation] | None = None,
    ) -> tuple[Draft, list[Citation]]:
        with self.tracer.span("generator", attempt=attempt, contexts=len(contexts), tier=tier.value) as span:
            if not contexts:
                return (
                    Draft(answer="", sentences=[], declined=True, attempt=attempt),
                    [],
                )

            prompt = prompts.generate_prompt(query, contexts)
            if extra_instruction:
                prompt += f"\n\nAdditional requirement for this attempt: {extra_instruction}"

            request = LLMRequest(
                task=Task.GENERATE,
                prompt=prompt,
                system=prompts.SYSTEM_BASE,
                payload={"query": query, "contexts": contexts, "attempt": attempt},
                tier=tier,
                max_tokens=2048,
            )

            try:
                response = await self.llm.complete(request)
                data = parse_json_lenient(response.text)
            except Exception as exc:  # noqa: BLE001
                if degradations is not None:
                    degradations.append(Degradation("generator", f"{type(exc).__name__}: {exc}"))
                span.attributes["declined"] = True
                return (
                    Draft(answer="", sentences=[], declined=True, attempt=attempt),
                    [],
                )

            if data.get("declined"):
                span.attributes["declined"] = True
                return (
                    Draft(
                        answer="",
                        sentences=[],
                        declined=True,
                        attempt=attempt,
                        model=response.model,
                    ),
                    [],
                )

            draft, citations = self._assemble(data, contexts, response.model, attempt)
            span.attributes["sentences"] = len(draft.sentences)
            span.attributes["citations"] = len(citations)
            return draft, citations

    def _assemble(
        self,
        data: dict[str, Any],
        contexts: list[dict[str, Any]],
        model: str,
        attempt: int,
    ) -> tuple[Draft, list[Citation]]:
        sentences: list[Sentence] = []
        used_blocks: list[int] = []

        for raw in data.get("sentences", []):
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            blocks: list[int] = []
            for value in raw.get("citations", []):
                try:
                    block = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= block <= len(contexts):
                    blocks.append(block)
            for block in blocks:
                if block not in used_blocks:
                    used_blocks.append(block)
            sentences.append(
                Sentence(text=text, citation_ids=[contexts[b - 1]["chunk_id"] for b in blocks])
            )

        # Renumber markers densely in order of first use, so the reader sees
        # [1][2][3] rather than the raw retrieval ranks.
        marker_for_block = {block: i + 1 for i, block in enumerate(used_blocks)}
        citations = [
            Citation(
                marker=f"[{marker_for_block[block]}]",
                chunk_id=contexts[block - 1]["chunk_id"],
                label=contexts[block - 1]["label"],
                chapter=contexts[block - 1]["chapter"],
                section=contexts[block - 1]["section"],
                page_start=contexts[block - 1]["page_start"],
                page_end=contexts[block - 1]["page_end"],
                quote=contexts[block - 1]["text"][:300],
            )
            for block in used_blocks
        ]

        chunk_to_marker = {c.chunk_id: c.marker for c in citations}
        rendered = [
            _attach_markers(
                sentence.text,
                "".join(
                    chunk_to_marker[cid]
                    for cid in sentence.citation_ids
                    if cid in chunk_to_marker
                ),
            )
            for sentence in sentences
        ]

        draft = Draft(
            answer=" ".join(rendered).strip(),
            sentences=sentences,
            cited_chunk_ids=[c.chunk_id for c in citations],
            model=model,
            attempt=attempt,
            declined=not sentences,
        )
        return draft, citations
