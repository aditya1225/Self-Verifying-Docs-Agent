"""MCP server (Ch. 10): expose the corpus to Claude Desktop or any MCP client.

Three tools, deliberately layered so a client can choose how much of the
pipeline it wants:

  search_corpus   - raw hybrid retrieval, returns passages with citations.
  ask_corpus      - the full agent, including the critic and its refusals.
  corpus_info     - what is indexed, so the client can tell the user.

`ask_corpus` is the interesting one in a demo: the client gets an answer that
has already been fact-checked against the sources, or an honest refusal.
"""

from __future__ import annotations

import asyncio
import json

from .agent.pipeline import Agent
from .index.store import CorpusIndex
from .reliability import CorpusNotIndexedError
from .retrieval.hybrid import hybrid_retrieve
from .tracing import Tracer

SERVER_NAME = "self-verifying-docs"

_index: CorpusIndex | None = None


def _get_index() -> CorpusIndex:
    global _index
    if _index is None:
        _index = CorpusIndex.load()
    return _index


def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(SERVER_NAME)

    @mcp.tool()
    async def search_corpus(query: str, top_k: int = 5) -> str:
        """Search the indexed book and return the most relevant passages with citations.

        Args:
            query: What to look for.
            top_k: How many passages to return (1-10).
        """
        index = _get_index()
        result = await hybrid_retrieve(
            index, [query], tracer=Tracer(), top_k=max(1, min(top_k, 10))
        )
        return json.dumps(
            {
                "query": query,
                "reranked": result.reranked,
                "passages": [
                    {
                        "source": sc.chunk.citation_label,
                        "chapter": sc.chunk.chapter,
                        "section": sc.chunk.section,
                        "pages": [sc.chunk.page_start, sc.chunk.page_end],
                        "score": round(sc.score, 4),
                        "text": sc.chunk.text[:1500],
                    }
                    for sc in result.chunks
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    async def ask_corpus(question: str, profile: str = "full") -> str:
        """Ask a question and get a cited, fact-checked answer - or an honest refusal.

        The answer has passed a faithfulness critic: every claim was checked
        against the retrieved passages. If the book does not support an answer,
        this declines rather than guessing.

        Args:
            question: The question to answer.
            profile: Pipeline profile - "full" (default), "agentic", "hybrid" or "naive".
        """
        agent = Agent(_get_index(), profile=profile)
        answer = await agent.answer(question)
        return json.dumps(
            {
                "status": answer.status.value,
                "answer": answer.answer,
                "citations": [
                    {"marker": c.marker, "source": c.label} for c in answer.citations
                ],
                "faithfulness": round(answer.critic.faithfulness, 3) if answer.critic else None,
                "route": answer.route.route.value if answer.route else None,
                "sub_questions": answer.plan.sub_questions if answer.plan else [],
                "latency_ms": round(answer.latency_ms, 1),
                "warnings": answer.warnings,
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    async def corpus_info() -> str:
        """Describe what is currently indexed and searchable."""
        try:
            index = _get_index()
        except CorpusNotIndexedError as exc:
            return json.dumps({"indexed": False, "error": str(exc)})

        chapters: dict[str, int] = {}
        for chunk in index.chunks:
            chapters[chunk.chapter] = chapters.get(chunk.chapter, 0) + 1
        return json.dumps(
            {
                "indexed": True,
                "document": index.meta.doc_id,
                "chunks": index.meta.n_chunks,
                "embedding_backend": index.meta.embedding_backend,
                "created_at": index.meta.created_at,
                "chapters": chapters,
            },
            ensure_ascii=False,
            indent=2,
        )

    return mcp


def main() -> None:
    """Run over stdio, which is what Claude Desktop launches."""
    try:
        _get_index()
    except CorpusNotIndexedError as exc:
        raise SystemExit(f"{exc}") from exc
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
