"""`svda` command line."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent.memory import ConversationMemory
from .agent.pipeline import PROFILES, Agent
from .agent.review_queue import ReviewQueue
from .config import get_settings
from .eval import ci_gate, report as report_mod
from .eval.golden import append_golden, load_golden, summarize
from .eval.runner import load_report, run_ablation, save_report
from .index.store import CorpusIndex
from .llm import backend_name
from .reliability import CorpusNotIndexedError
from .schemas import AgentAnswer, AnswerStatus

app = typer.Typer(
    add_completion=False,
    help="Self-Verifying Docs Agent - agentic RAG with citations, reflection and a CI gate.",
)
golden_app = typer.Typer(help="Inspect and grow the golden set.")
review_app = typer.Typer(help="Human-in-the-loop review queue.")
app.add_typer(golden_app, name="golden")
app.add_typer(review_app, name="review")

console = Console()

STATUS_STYLE = {
    AnswerStatus.ANSWERED: "green",
    AnswerStatus.DECLINED: "yellow",
    AnswerStatus.BLOCKED: "red",
    AnswerStatus.HUMAN_REVIEW: "magenta",
}


def _load_index() -> CorpusIndex:
    try:
        return CorpusIndex.load()
    except CorpusNotIndexedError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)


def _print_answer(answer: AgentAnswer, show_trace: bool = False) -> None:
    style = STATUS_STYLE.get(answer.status, "white")
    console.print(
        Panel(
            answer.answer or "(no answer)",
            title=f"[{style}]{answer.status.value}[/{style}]",
            subtitle=f"profile={answer.profile}  {answer.latency_ms:.0f} ms  "
            f"${answer.usage.cost_usd:.5f}  trace={answer.trace_id}",
            border_style=style,
        )
    )

    if answer.citations:
        table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        table.add_column("")
        table.add_column("Source")
        for citation in answer.citations:
            table.add_row(citation.marker, citation.label)
        console.print(table)

    if answer.route:
        console.print(
            f"[dim]route={answer.route.route.value} ({answer.route.decided_by}, "
            f"conf {answer.route.confidence:.2f}) - {answer.route.reason}[/dim]"
        )
    if answer.plan and answer.plan.is_decomposed:
        for sub in answer.plan.sub_questions:
            console.print(f"[dim]  hop: {sub}[/dim]")
    if answer.critic:
        console.print(
            f"[dim]critic={answer.critic.action.value} "
            f"faithfulness={answer.critic.faithfulness:.0%} "
            f"citation_coverage={answer.critic.citation_coverage:.0%}[/dim]"
        )
    for warning in answer.warnings:
        console.print(f"[yellow]degraded:[/yellow] [dim]{warning}[/dim]")
    if show_trace:
        for name, ms in sorted(answer.stage_latency_ms.items(), key=lambda kv: -kv[1]):
            console.print(f"[dim]  {name}: {ms:.1f} ms[/dim]")


# --------------------------------------------------------------------------


@app.command()
def ingest(
    pdf: Path = typer.Option(..., "--pdf", help="Path to the source PDF."),
    doc_id: Optional[str] = typer.Option(None, help="Identifier for this document."),
    backend: Optional[str] = typer.Option(None, help="Embedding backend: local | hash."),
    all_sections: bool = typer.Option(False, help="Include front/back matter too."),
) -> None:
    """Ingest a PDF: chunk it, guard it, embed it, index it."""
    from .ingest.pipeline import ingest_pdf

    with console.status(f"Ingesting {pdf.name} ..."):
        _, report = ingest_pdf(
            pdf, doc_id=doc_id, embedding_backend=backend, body_only=not all_sections
        )
    console.print(f"[green]{report.summary()}[/green]")
    if report.quarantine_reasons:
        console.print("[yellow]Quarantined by the document guardrail:[/yellow]")
        for reason in report.quarantine_reasons:
            console.print(f"  [dim]{reason}[/dim]")
    console.print(f"[dim]index written to {get_settings().index_dir}[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the corpus."),
    profile: str = typer.Option("full", help=f"One of: {', '.join(PROFILES)}"),
    as_json: bool = typer.Option(False, "--json", help="Emit the full AgentAnswer as JSON."),
    trace: bool = typer.Option(False, help="Show per-stage latency."),
) -> None:
    """Ask one question."""
    if profile not in PROFILES:
        console.print(f"[red]unknown profile {profile!r}; choose from {list(PROFILES)}[/red]")
        raise typer.Exit(code=2)

    index = _load_index()
    agent = Agent(index, profile=profile, review_queue=ReviewQueue())
    answer = asyncio.run(agent.answer(question))

    if as_json:
        console.print_json(answer.model_dump_json())
    else:
        _print_answer(answer, show_trace=trace)


@app.command()
def chat(profile: str = typer.Option("full", help="Pipeline profile.")) -> None:
    """Interactive session with conversational memory (Ch. 8)."""
    index = _load_index()
    memory = ConversationMemory()
    agent = Agent(index, profile=profile, memory=memory, review_queue=ReviewQueue())
    console.print(f"[dim]backend={backend_name()}  profile={profile}  (ctrl-c to exit)[/dim]")
    while True:
        try:
            question = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not question:
            continue
        if question in {"/exit", "/quit"}:
            return
        if question == "/clear":
            memory.clear()
            console.print("[dim]memory cleared[/dim]")
            continue
        _print_answer(asyncio.run(agent.answer(question)))


@app.command("eval")
def eval_cmd(
    profiles: Optional[str] = typer.Option(
        None, help="Comma-separated profiles. Default: naive,hybrid,agentic,full"
    ),
    limit: Optional[int] = typer.Option(None, help="Only the first N golden items."),
    concurrency: int = typer.Option(4, help="Parallel questions per profile."),
    llm_judge: bool = typer.Option(
        False, help="Grade faithfulness with a separate LLM judge instead of the offline check."
    ),
    dashboard: bool = typer.Option(True, help="Write runs/dashboard.html."),
    baseline: bool = typer.Option(False, help="Record this run as the new CI baseline."),
) -> None:
    """Run the ablation over the golden set and write the report."""
    index = _load_index()
    names = [p.strip() for p in profiles.split(",")] if profiles else None
    items = load_golden()
    n_items = min(limit, len(items)) if limit else len(items)
    n_profiles = len(names) if names else 4

    console.print(
        f"[dim]{n_items} questions x {n_profiles} profiles = "
        f"{n_items * n_profiles} runs, backend={backend_name()}[/dim]"
    )

    with console.status("Running ablation ...") as status:
        done = {"n": 0}

        def progress() -> None:
            done["n"] += 1
            status.update(f"Running ablation ... {done['n']}/{n_items * n_profiles}")

        report = asyncio.run(
            run_ablation(
                index,
                profiles=names,
                limit=limit,
                concurrency=concurrency,
                use_llm_judge=llm_judge,
                progress=progress,
            )
        )

    path = save_report(report)
    console.print()
    console.print(report_mod.ablation_markdown(report))
    console.print()
    console.print(f"[dim]report: {path}[/dim]")

    md_path = report_mod.write_markdown(report)
    console.print(f"[dim]markdown: {md_path}[/dim]")
    if dashboard:
        console.print(f"[dim]dashboard: {report_mod.write_dashboard(report)}[/dim]")
    if baseline:
        console.print(f"[green]baseline updated: {report_mod.write_baseline(report)}[/green]")


@app.command()
def gate(
    report_path: Optional[Path] = typer.Option(None, "--report", help="Defaults to runs/latest.json."),
    profile: str = typer.Option("full", help="Profile to gate on."),
    tolerance: float = typer.Option(ci_gate.DEFAULT_TOLERANCE, help="Allowed regression."),
) -> None:
    """CI gate: exit non-zero when quality floors or the baseline are breached."""
    path = report_path or get_settings().runs_dir / "latest.json"
    if not path.exists():
        console.print(f"[red]no eval report at {path}; run `svda eval` first[/red]")
        raise typer.Exit(code=2)

    result = ci_gate.check(load_report(path), profile=profile, tolerance=tolerance)
    console.print(("[green]" if result.passed else "[red]") + result.report() + "[/]")
    raise typer.Exit(code=0 if result.passed else 1)


@app.command()
def dashboard(
    report_path: Optional[Path] = typer.Option(None, "--report"),
) -> None:
    """Regenerate runs/dashboard.html from the latest eval report and traces."""
    path = report_path or get_settings().runs_dir / "latest.json"
    if not path.exists():
        console.print(f"[red]no eval report at {path}; run `svda eval` first[/red]")
        raise typer.Exit(code=2)
    console.print(f"[green]{report_mod.write_dashboard(load_report(path))}[/green]")


@app.command()
def verify() -> None:
    """Self-verification: run the offline invariant suite (no API key needed)."""
    import subprocess
    import sys

    root = get_settings().project_root
    console.print("[dim]running pytest ...[/dim]")
    code = subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=root)
    raise typer.Exit(code=code)


@app.command()
def mcp() -> None:
    """Run the MCP server so Claude Desktop can query this corpus (Ch. 10)."""
    from .mcp_server import main as mcp_main

    mcp_main()


@app.command()
def info() -> None:
    """Show the current configuration and index state."""
    settings = get_settings()
    table = Table(show_header=False, box=None)
    table.add_row("LLM backend", backend_name())
    table.add_row("Strong / cheap model", f"{settings.model_strong} / {settings.model_cheap}")
    table.add_row("Embedding backend", settings.embedding_backend)
    table.add_row("Index dir", str(settings.index_dir))
    try:
        index = CorpusIndex.load()
        table.add_row("Indexed document", f"{index.meta.doc_id} ({index.meta.n_chunks} chunks)")
        table.add_row("Index embeddings", index.meta.embedding_backend)
        table.add_row("Indexed at", index.meta.created_at)
    except CorpusNotIndexedError:
        table.add_row("Indexed document", "[yellow]none - run `svda ingest`[/yellow]")
    counts = summarize(load_golden()) if settings.golden_path.exists() else {}
    if counts:
        table.add_row("Golden set", ", ".join(f"{k}={v}" for k, v in counts.items()))
    console.print(table)


# -- golden ----------------------------------------------------------------


@golden_app.command("stats")
def golden_stats() -> None:
    """Counts by question kind."""
    counts = summarize(load_golden())
    for key, value in counts.items():
        console.print(f"{key:>14}: {value}")


# -- review ----------------------------------------------------------------


@review_app.command("list")
def review_list(pending_only: bool = typer.Option(True, "--pending/--all")) -> None:
    """Show queued low-confidence answers."""
    queue = ReviewQueue()
    items = queue.pending() if pending_only else queue.load()
    if not items:
        console.print("[dim]review queue is empty[/dim]")
        return
    table = Table()
    table.add_column("id")
    table.add_column("question")
    table.add_column("faith", justify="right")
    table.add_column("reason")
    for item in items:
        table.add_row(item.id, item.query[:60], f"{item.faithfulness:.0%}", item.reason[:40])
    console.print(table)


@review_app.command("verdict")
def review_verdict(
    item_id: str = typer.Argument(...),
    verdict: str = typer.Argument(..., help="accept | reject | corrected"),
    answer: str = typer.Option("", help="Corrected answer text, for `corrected`."),
) -> None:
    """Record a reviewer decision."""
    if ReviewQueue().record_verdict(item_id, verdict, answer):
        console.print(f"[green]{item_id} marked {verdict}[/green]")
    else:
        console.print(f"[red]no queue item {item_id}[/red]")
        raise typer.Exit(code=1)


@review_app.command("promote")
def review_promote() -> None:
    """Add reviewed items to the golden set, closing the eval loop (Ch. 13)."""
    items = ReviewQueue().to_golden_items()
    added = append_golden(items)
    console.print(f"[green]promoted {added} reviewed item(s) into the golden set[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
