"""Central configuration. Every knob the pipeline reads lives here."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Immutable runtime settings, resolved from the environment once."""

    # --- paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    index_dir: Path = PROJECT_ROOT / "index"
    runs_dir: Path = PROJECT_ROOT / "runs"
    golden_path: Path = PROJECT_ROOT / "data" / "golden" / "golden_set.jsonl"
    baseline_path: Path = PROJECT_ROOT / "data" / "baseline_metrics.json"
    review_queue_path: Path = PROJECT_ROOT / "runs" / "review_queue.jsonl"

    # --- models (Ch. 16 resource-aware optimization) ------------------------
    model_strong: str = Field(default_factory=lambda: os.getenv("SVDA_MODEL_STRONG", "claude-opus-5"))
    model_cheap: str = Field(default_factory=lambda: os.getenv("SVDA_MODEL_CHEAP", "claude-haiku-4-5"))
    force_stub_llm: bool = Field(default_factory=lambda: _env_bool("SVDA_FORCE_STUB_LLM"))

    # Published Anthropic list prices, USD per 1M tokens. Used for cost accounting.
    price_table: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "claude-opus-5": (5.00, 25.00),
            "claude-sonnet-5": (2.00, 10.00),
            "claude-haiku-4-5": (1.00, 5.00),
            "stub": (0.0, 0.0),
        }
    )

    # --- ingest / chunking --------------------------------------------------
    chunk_target_chars: int = 1400
    chunk_overlap_chars: int = 200
    min_chunk_chars: int = 250

    # --- retrieval ----------------------------------------------------------
    embedding_backend: str = Field(default_factory=lambda: os.getenv("SVDA_EMBEDDING_BACKEND", "local"))
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_top_k: int = 25
    vector_top_k: int = 25
    rrf_k: int = 60
    rerank_top_k: int = 8
    final_context_chunks: int = 6

    # --- agent behaviour ----------------------------------------------------
    max_sub_questions: int = 4
    max_regeneration_attempts: int = 1
    faithfulness_accept_threshold: float = 0.80
    human_review_threshold: float = 0.55
    memory_turns: int = 6

    # --- exception handling (Ch. 12) ---------------------------------------
    llm_timeout_s: float = 90.0
    retrieval_timeout_s: float = 30.0
    rerank_timeout_s: float = 20.0
    max_retries: int = 2

    @property
    def anthropic_api_key(self) -> str | None:
        key = os.getenv("ANTHROPIC_API_KEY")
        return key.strip() or None if key else None

    def price_for(self, model: str) -> tuple[float, float]:
        """(input $/1M, output $/1M); unknown models are priced as the strong tier."""
        return self.price_table.get(model, self.price_table.get(self.model_strong, (5.0, 25.0)))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
