"""Golden-set loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import get_settings
from ..schemas import GoldenItem


def load_golden(path: Path | None = None) -> list[GoldenItem]:
    path = path or get_settings().golden_path
    if not path.exists():
        raise FileNotFoundError(f"golden set not found at {path}")
    items: list[GoldenItem] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(GoldenItem.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path}:{lineno} is not a valid golden item: {exc}") from exc
    _validate(items)
    return items


def append_golden(items: list[GoldenItem], path: Path | None = None) -> int:
    """Append new items, skipping ids that already exist."""
    path = path or get_settings().golden_path
    existing = {item.id for item in (load_golden(path) if path.exists() else [])}
    new = [item for item in items if item.id not in existing]
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in new:
            fh.write(item.model_dump_json() + "\n")
    return len(new)


def _validate(items: list[GoldenItem]) -> None:
    ids = [item.id for item in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate golden ids: {sorted(duplicates)}")
    for item in items:
        if item.answerable and item.kind.value not in {"chitchat"} and not item.answer_keywords:
            raise ValueError(f"{item.id}: answerable items need answer_keywords")
        if not item.answerable and item.answer_keywords:
            raise ValueError(f"{item.id}: unanswerable items must not have answer_keywords")


def summarize(items: list[GoldenItem]) -> dict[str, int]:
    counts: dict[str, int] = {"total": len(items)}
    for item in items:
        counts[item.kind.value] = counts.get(item.kind.value, 0) + 1
    counts["unanswerable"] = sum(1 for i in items if not i.answerable)
    return counts


def write_golden(items: list[GoldenItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
