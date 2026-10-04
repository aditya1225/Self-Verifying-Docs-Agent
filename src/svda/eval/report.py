"""Ablation table + monitoring dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path

from ..config import get_settings
from ..schemas import EvalReport, ProfileMetrics
from ..tracing import load_spans

PROFILE_LABELS = {
    "naive": "Naive RAG",
    "hybrid": "+ Hybrid + re-rank",
    "agentic": "+ Routing + planning",
    "full": "+ Reflection (full)",
}

COLUMNS: list[tuple[str, str, str]] = [
    # (attribute, header, format)
    ("answer_accuracy", "Answer acc.", "pct"),
    ("faithfulness", "Faithfulness", "pct"),
    ("citation_coverage", "Citation cov.", "pct"),
    ("multihop_accuracy", "Multi-hop acc.", "pct"),
    ("retrieval_recall", "Retr. recall", "pct"),
    ("retrieval_precision", "Retr. prec.", "pct"),
    ("refusal_correctness", "Correct refusals", "pct"),
    ("mean_latency_ms", "Mean lat.", "ms"),
    ("p95_latency_ms", "P95 lat.", "ms"),
    ("mean_cost_usd", "Cost/query", "usd"),
]


def _fmt(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:.0f} ms"
    if kind == "usd":
        return f"${value:.5f}"
    return str(value)


def ablation_markdown(report: EvalReport) -> str:
    headers = ["Profile"] + [h for _, h, _ in COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for metrics in report.profiles:
        row = [PROFILE_LABELS.get(metrics.profile, metrics.profile)]
        row += [_fmt(getattr(metrics, attr), kind) for attr, _, kind in COLUMNS]
        lines.append("| " + " | ".join(row) + " |")

    out = [
        f"### Ablation - {report.profiles[0].n if report.profiles else 0} golden questions",
        f"_run `{report.run_id}` at {report.created_at}, backend `{report.llm_backend}`_",
        "",
        *lines,
    ]

    delta = _headline_delta(report)
    if delta:
        out += ["", delta]
    return "\n".join(out)


def _headline_delta(report: EvalReport) -> str:
    """The one sentence a hiring manager actually reads."""
    before = report.metrics_for("agentic")
    after = report.metrics_for("full")
    if not before or not after:
        return ""
    faith_delta = (after.faithfulness - before.faithfulness) * 100
    latency_delta = after.mean_latency_ms - before.mean_latency_ms
    cost_delta = after.mean_cost_usd - before.mean_cost_usd
    return (
        f"**Headline:** reflection moved faithfulness from "
        f"{before.faithfulness * 100:.1f}% to {after.faithfulness * 100:.1f}% "
        f"({faith_delta:+.1f} pts) for {latency_delta:+.0f} ms and "
        f"${cost_delta:+.5f} per query."
    )


# --------------------------------------------------------------------------
# HTML dashboard
# --------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; --bg:#fbfbfa; --fg:#1b1b1a; --muted:#6b6b68;
  --line:#e3e3e0; --card:#ffffff; --accent:#1a6b5a; --warn:#9a5b16; --bad:#9b2c2c; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181a; --fg:#e9e9e6;
  --muted:#9a9a96; --line:#2c2f33; --card:#1d2023; --accent:#5fcdb4;
  --warn:#e0a35c; --bad:#e2756f; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.55
  -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:22px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:15px; margin:34px 0 12px; text-transform:uppercase;
  letter-spacing:0.07em; color:var(--muted); font-weight:600; }
.sub { color:var(--muted); margin:0 0 8px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th,td { padding:9px 12px; text-align:right; border-bottom:1px solid var(--line);
  white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
thead th { font-size:11px; text-transform:uppercase; letter-spacing:0.05em;
  color:var(--muted); font-weight:600; }
tbody tr:last-child td { border-bottom:none; }
tbody tr:last-child { font-weight:600; }
.best { color:var(--accent); }
.grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); }
.tile .k { font-size:11px; text-transform:uppercase; letter-spacing:0.05em;
  color:var(--muted); }
.tile .v { font-size:26px; font-weight:600; letter-spacing:-0.02em; margin-top:2px; }
.tile .d { font-size:12px; color:var(--muted); margin-top:2px; }
.bar { height:7px; background:var(--line); border-radius:4px; overflow:hidden; min-width:90px; }
.bar > i { display:block; height:100%; background:var(--accent); }
.pill { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11px;
  border:1px solid var(--line); color:var(--muted); }
.note { color:var(--muted); font-size:12.5px; margin-top:10px; }
code { background:var(--line); padding:1px 5px; border-radius:4px; font-size:12px; }
"""


def _tile(key: str, value: str, detail: str = "") -> str:
    return (
        f'<div class="card tile"><div class="k">{html.escape(key)}</div>'
        f'<div class="v">{html.escape(value)}</div>'
        f'<div class="d">{html.escape(detail)}</div></div>'
    )


def _ablation_table_html(report: EvalReport) -> str:
    best: dict[str, float] = {}
    for attr, _, kind in COLUMNS:
        values = [getattr(m, attr) for m in report.profiles]
        if not values:
            continue
        best[attr] = min(values) if kind in {"ms", "usd"} else max(values)

    head = "".join(f"<th>{html.escape(h)}</th>" for _, h, _ in COLUMNS)
    rows = []
    for metrics in report.profiles:
        cells = []
        for attr, _, kind in COLUMNS:
            value = getattr(metrics, attr)
            cls = ' class="best"' if best.get(attr) == value else ""
            cells.append(f"<td{cls}>{html.escape(_fmt(value, kind))}</td>")
        label = PROFILE_LABELS.get(metrics.profile, metrics.profile)
        rows.append(f"<tr><td>{html.escape(label)}</td>{''.join(cells)}</tr>")

    return (
        '<div class="card scroll"><table><thead><tr><th>Profile</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _stage_table_html(spans: list[dict]) -> str:
    if not spans:
        return '<div class="card"><p class="sub">No traces recorded yet.</p></div>'

    agg: dict[str, list[float]] = {}
    for span in spans:
        agg.setdefault(span.get("name", "?"), []).append(float(span.get("duration_ms", 0.0)))

    ordered = sorted(agg.items(), key=lambda kv: -sum(kv[1]))
    worst = max((sum(v) / len(v)) for v in agg.values()) or 1.0

    rows = []
    for name, values in ordered[:14]:
        avg = sum(values) / len(values)
        width = max(2, round(avg / worst * 100))
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{len(values)}</td>"
            f"<td>{avg:.1f} ms</td><td>{max(values):.1f} ms</td>"
            f'<td><div class="bar"><i style="width:{width}%"></i></div></td></tr>'
        )
    return (
        '<div class="card scroll"><table><thead><tr><th>Stage</th><th>Calls</th>'
        "<th>Mean</th><th>Max</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _decisions_html(spans: list[dict]) -> str:
    routes: dict[str, int] = {}
    critic: dict[str, int] = {}
    for span in spans:
        attrs = span.get("attributes", {}) or {}
        if span.get("name") == "router" and attrs.get("route"):
            routes[str(attrs["route"])] = routes.get(str(attrs["route"]), 0) + 1
        if span.get("name") == "critic" and attrs.get("action"):
            critic[str(attrs["action"])] = critic.get(str(attrs["action"]), 0) + 1

    def block(title: str, counts: dict[str, int]) -> str:
        if not counts:
            return ""
        total = sum(counts.values())
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v}</td><td>{v / total * 100:.0f}%</td></tr>"
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        return (
            f'<div class="card scroll"><h2 style="margin-top:0">{html.escape(title)}</h2>'
            f"<table><tbody>{rows}</tbody></table></div>"
        )

    blocks = block("Routing decisions", routes) + block("Critic verdicts", critic)
    return blocks or '<div class="card"><p class="sub">No decisions traced yet.</p></div>'


def render_dashboard(report: EvalReport, spans: list[dict] | None = None) -> str:
    spans = spans if spans is not None else load_spans()
    full = report.metrics_for("full") or (report.profiles[-1] if report.profiles else ProfileMetrics(profile="n/a"))
    naive = report.metrics_for("naive")

    def delta(attr: str) -> str:
        if not naive:
            return ""
        diff = (getattr(full, attr) - getattr(naive, attr)) * 100
        return f"{diff:+.1f} pts vs naive"

    tiles = "".join(
        [
            _tile("Faithfulness", _fmt(full.faithfulness, "pct"), delta("faithfulness")),
            _tile("Answer accuracy", _fmt(full.answer_accuracy, "pct"), delta("answer_accuracy")),
            _tile("Citation coverage", _fmt(full.citation_coverage, "pct"), delta("citation_coverage")),
            _tile("Multi-hop accuracy", _fmt(full.multihop_accuracy, "pct"), delta("multihop_accuracy")),
            _tile("Correct refusals", _fmt(full.refusal_correctness, "pct"), delta("refusal_correctness")),
            _tile("P95 latency", _fmt(full.p95_latency_ms, "ms"), f"mean {_fmt(full.mean_latency_ms, 'ms')}"),
            _tile("Cost / query", _fmt(full.mean_cost_usd, "usd"), f"total {_fmt(full.total_cost_usd, 'usd')}"),
            _tile("Golden items", str(full.n), f"{full.errors} errors"),
        ]
    )

    headline = _headline_delta(report)
    headline_html = (
        f'<p class="note">{html.escape(headline.replace("**", ""))}</p>' if headline else ""
    )

    return f"""<title>SVDA Evaluation Dashboard</title>
<style>{_CSS}</style>
<div class="wrap">
  <h1>Self-Verifying Docs Agent - evaluation</h1>
  <p class="sub">Run <code>{html.escape(report.run_id)}</code> &middot;
     {html.escape(report.created_at)} &middot;
     backend <span class="pill">{html.escape(report.llm_backend)}</span></p>

  <h2>Headline (full pipeline)</h2>
  <div class="grid">{tiles}</div>

  <h2>Ablation</h2>
  {_ablation_table_html(report)}
  {headline_html}

  <h2>Stage latency (from traces)</h2>
  {_stage_table_html(spans)}

  <h2>Decisions</h2>
  <div class="grid">{_decisions_html(spans)}</div>

  <p class="note">Faithfulness is graded independently of the pipeline's own critic,
  so the <em>naive</em> profile (which has no critic) is measured on the same scale
  as the <em>full</em> one.</p>
</div>"""


def write_dashboard(report: EvalReport, path: Path | None = None) -> Path:
    path = path or get_settings().runs_dir / "dashboard.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(report), encoding="utf-8")
    return path


def write_markdown(report: EvalReport, path: Path | None = None) -> Path:
    path = path or get_settings().runs_dir / "ablation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ablation_markdown(report) + "\n", encoding="utf-8")
    return path


def write_baseline(report: EvalReport, path: Path | None = None) -> Path:
    """Freeze the current `full`-profile metrics as the CI baseline."""
    path = path or get_settings().baseline_path
    metrics = report.metrics_for("full")
    if metrics is None:
        raise ValueError("report has no `full` profile to baseline")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": report.run_id,
                "created_at": report.created_at,
                "llm_backend": report.llm_backend,
                "metrics": metrics.model_dump(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
