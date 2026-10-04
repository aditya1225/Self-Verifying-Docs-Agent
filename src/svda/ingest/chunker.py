"""Section-aware chunking.

Chunks never span a section boundary, so a citation always names one real
place in the book. Within a section we pack paragraphs up to a target size
with a small overlap, and we track which pages each chunk drew from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import get_settings
from ..schemas import Chunk
from .pdf_loader import Page, SectionSpan


@dataclass
class _Block:
    text: str
    page: int


def _blocks_for_span(span: SectionSpan, pages_by_number: dict[int, Page]) -> list[_Block]:
    blocks: list[_Block] = []
    for page_no in range(span.start_page, span.end_page + 1):
        page = pages_by_number.get(page_no)
        if page is None or not page.text:
            continue
        for para in re.split(r"\n\s*\n", page.text):
            para = para.strip()
            if len(para) < 40:
                continue
            blocks.append(_Block(text=para, page=page_no))
    return blocks


def chunk_sections(
    spans: list[SectionSpan],
    pages: list[Page],
    doc_id: str,
) -> list[Chunk]:
    settings = get_settings()
    pages_by_number = {p.number: p for p in pages}
    chunks: list[Chunk] = []
    ordinal = 0

    for span in spans:
        blocks = _blocks_for_span(span, pages_by_number)
        if not blocks:
            continue

        buffer: list[_Block] = []
        buffer_len = 0

        def flush() -> None:
            nonlocal buffer, buffer_len, ordinal
            if not buffer:
                return
            text = "\n\n".join(b.text for b in buffer).strip()
            if len(text) >= settings.min_chunk_chars:
                ordinal += 1
                chunks.append(
                    Chunk(
                        id=f"{doc_id}::c{ordinal:05d}",
                        doc_id=doc_id,
                        text=text,
                        chapter=span.chapter,
                        chapter_number=span.chapter_number,
                        section=span.section,
                        page_start=min(b.page for b in buffer),
                        page_end=max(b.page for b in buffer),
                        ordinal=ordinal,
                    )
                )
            # Carry the tail forward as overlap so a fact split across the
            # boundary is still retrievable from one chunk.
            overlap: list[_Block] = []
            overlap_len = 0
            for block in reversed(buffer):
                if overlap_len >= settings.chunk_overlap_chars:
                    break
                overlap.insert(0, block)
                overlap_len += len(block.text)
            buffer = overlap if overlap_len < settings.chunk_target_chars else []
            buffer_len = sum(len(b.text) for b in buffer)

        for block in blocks:
            # A single oversized paragraph becomes its own chunk, split on
            # sentence boundaries if it is truly huge.
            if len(block.text) > settings.chunk_target_chars * 2:
                flush()
                for piece in _split_long(block.text, settings.chunk_target_chars):
                    ordinal += 1
                    chunks.append(
                        Chunk(
                            id=f"{doc_id}::c{ordinal:05d}",
                            doc_id=doc_id,
                            text=piece,
                            chapter=span.chapter,
                            chapter_number=span.chapter_number,
                            section=span.section,
                            page_start=block.page,
                            page_end=block.page,
                            ordinal=ordinal,
                        )
                    )
                buffer, buffer_len = [], 0
                continue

            buffer.append(block)
            buffer_len += len(block.text)
            if buffer_len >= settings.chunk_target_chars:
                flush()

        flush()

    return chunks


def _split_long(text: str, target: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        current.append(sentence)
        length += len(sentence)
        if length >= target:
            out.append(" ".join(current).strip())
            current, length = [], 0
    if current:
        tail = " ".join(current).strip()
        if tail:
            out.append(tail)
    return [piece for piece in out if len(piece) >= 100]
