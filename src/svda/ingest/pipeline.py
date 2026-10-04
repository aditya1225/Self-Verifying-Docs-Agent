"""ingest -> chunk -> embed -> index, with a document guardrail in the middle.

Ingested text is untrusted input: a PDF can carry an injected instruction just
as easily as a user can type one. Chunks that fail the document guardrail are
quarantined rather than indexed (Ch. 18).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..index.embeddings import build_embedder
from ..index.store import CorpusIndex, IndexMeta
from ..schemas import Chunk
from .chunker import chunk_sections
from .pdf_loader import is_body_section, load_pages, load_section_map


@dataclass
class IngestReport:
    doc_id: str
    source_path: str
    pages: int
    sections: int
    chunks: int
    quarantined: int
    embedding_backend: str
    quarantine_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.doc_id}: {self.pages} pages -> {self.sections} sections -> "
            f"{self.chunks} chunks ({self.quarantined} quarantined) "
            f"via {self.embedding_backend}"
        )


def ingest_pdf(
    pdf_path: Path,
    *,
    doc_id: str | None = None,
    index_dir: Path | None = None,
    embedding_backend: str | None = None,
    body_only: bool = True,
) -> tuple[CorpusIndex, IngestReport]:
    from ..agent.guardrails import scan_document

    settings = get_settings()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    doc_id = doc_id or pdf_path.stem[:40]

    pages = load_pages(pdf_path)
    spans = load_section_map(pdf_path, total_pages=len(pages))
    if body_only:
        spans = [s for s in spans if is_body_section(s)]

    raw_chunks = chunk_sections(spans, pages, doc_id=doc_id)

    kept: list[Chunk] = []
    quarantine_reasons: list[str] = []
    for chunk in raw_chunks:
        verdict = scan_document(chunk.text)
        if verdict.passed:
            kept.append(chunk)
        else:
            quarantine_reasons.append(
                f"{chunk.id} ({chunk.citation_label}): "
                + "; ".join(v.detail or v.rule for v in verdict.blocking)
            )

    # Re-number so chunk ids stay dense and ordered after quarantine.
    for i, chunk in enumerate(kept, start=1):
        chunk.ordinal = i

    embedder = build_embedder(embedding_backend)
    vectors = (
        embedder.encode([c.text for c in kept])
        if kept
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )

    meta = IndexMeta(
        doc_id=doc_id,
        source_path=str(pdf_path),
        embedding_backend=embedder.name,
        embedding_dim=embedder.dim,
        n_chunks=len(kept),
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    index = CorpusIndex(kept, vectors, meta)
    index.save(index_dir or settings.index_dir)

    report = IngestReport(
        doc_id=doc_id,
        source_path=str(pdf_path),
        pages=len(pages),
        sections=len(spans),
        chunks=len(kept),
        quarantined=len(quarantine_reasons),
        embedding_backend=embedder.name,
        quarantine_reasons=quarantine_reasons[:20],
    )
    return index, report


def build_index_from_chunks(
    chunks: list[Chunk],
    *,
    doc_id: str = "test",
    embedding_backend: str | None = "hash",
) -> CorpusIndex:
    """In-memory index for tests and fixtures."""
    embedder = build_embedder(embedding_backend)
    vectors = (
        embedder.encode([c.text for c in chunks])
        if chunks
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )
    meta = IndexMeta(
        doc_id=doc_id,
        source_path="<memory>",
        embedding_backend=embedder.name,
        embedding_dim=embedder.dim,
        n_chunks=len(chunks),
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    return CorpusIndex(chunks, vectors, meta)
