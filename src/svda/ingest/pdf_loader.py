"""PDF -> clean pages + a chapter/section map derived from the PDF outline.

The outline is what gives citations their value: instead of "chunk 412" the
agent can say "Chapter 8: Transactions > Weak Isolation Levels, pp. 311-313".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\.\s*(.+)$", re.IGNORECASE)
# Lines that are page furniture rather than prose.
_NOISE_LINE_RE = re.compile(
    r"^\s*(?:\d{1,4}|[ivxlc]{1,7}|\|)\s*$",
    re.IGNORECASE,
)


@dataclass
class Page:
    number: int  # 1-based PDF page number
    text: str


@dataclass
class SectionSpan:
    chapter_number: int | None
    chapter: str
    section: str
    start_page: int  # 1-based, inclusive
    end_page: int  # 1-based, inclusive


def load_pages(pdf_path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [Page(number=i + 1, text=clean_page_text(p.extract_text() or "")) for i, p in enumerate(reader.pages)]


def clean_page_text(raw: str) -> str:
    """Drop running headers/footers and re-join words broken across lines."""
    lines = [ln.rstrip() for ln in raw.splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            kept.append("")
            continue
        if _NOISE_LINE_RE.match(line):
            continue
        stripped = line.strip()
        # Running heads: "Chapter 6: Replication", "262 | Chapter 6: Replication"
        if re.match(r"^\d{0,4}\s*\|?\s*Chapter\s+\d+:", stripped):
            continue
        if re.match(r"^(Summary|References)\s*\|\s*\d+$", stripped):
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_section_map(pdf_path: Path, total_pages: int) -> list[SectionSpan]:
    """Flatten the PDF outline into non-overlapping (chapter, section) spans."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    try:
        outline = reader.outline
    except Exception:  # noqa: BLE001 - some PDFs have no usable outline
        outline = []

    entries: list[tuple[int, int, str]] = []  # (depth, page, title)

    def walk(items, depth: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:  # noqa: BLE001
                continue
            title = str(getattr(item, "title", "") or "").strip()
            if title:
                entries.append((depth, page, title))

    walk(outline)
    entries.sort(key=lambda e: e[1])

    if not entries:
        return [SectionSpan(None, "Document", "Full text", 1, total_pages)]

    spans: list[SectionSpan] = []
    current_chapter_num: int | None = None
    current_chapter = "Front Matter"

    for i, (depth, page, title) in enumerate(entries):
        match = _CHAPTER_RE.match(title)
        if match:
            current_chapter_num = int(match.group(1))
            current_chapter = f"Chapter {current_chapter_num}: {match.group(2).strip()}"
            section = "Introduction"
        elif depth == 0:
            current_chapter_num = None
            current_chapter = title
            section = title
        else:
            section = title

        end_page = entries[i + 1][1] - 1 if i + 1 < len(entries) else total_pages
        if end_page < page:
            end_page = page
        spans.append(
            SectionSpan(
                chapter_number=current_chapter_num,
                chapter=current_chapter,
                section=section,
                start_page=page,
                end_page=end_page,
            )
        )

    return spans


# Front/back matter carries no answerable content but a lot of index noise.
_EXCLUDED_CHAPTER_TITLES = {
    "cover", "copyright", "table of contents", "index", "colophon",
    "about the authors", "praise for designing data-intensive applications",
}


def is_body_section(span: SectionSpan) -> bool:
    title = span.chapter.strip().lower()
    if title in _EXCLUDED_CHAPTER_TITLES:
        return False
    return span.chapter_number is not None or title in {"preface", "glossary"}
