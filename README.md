# Self-Verifying Docs Agent

Agentic RAG over *Designing Data-Intensive Applications* (2nd ed., 673 pages) that
**cites every claim, checks its own answers against the sources, and declines when the
corpus doesn't support one.**

The point of the project isn't that it uses agentic patterns — it's that each pattern is
switchable, so the cost of each one is *measured* rather than asserted:

```
Query → Guardrails (input) → Router → Planner (decompose if multi-hop)
      → Parallel hybrid retrieval (BM25 + vector) → Re-ranker
      → Generator (with citations) → Reflection/Critic (faithfulness check)
      → Guardrails (output) → Answer | "I can't support this" | Human review

Every step traced → evaluation dashboard + CI gate
```

```
$ svda ask "How do quorums in leaderless replication interact with linearizability?"

╭─ answered ───────────────────────────────────────────────────────────────────╮
│ A quorum read and write does not by itself provide linearizability, because  │
│ concurrent operations can still observe stale values [1]. Linearizability is │
│ a recency guarantee: it makes a system appear as if there were only one copy │
│ of the data [1].                                                             │
╰─ profile=full  1483 ms  $0.00412  trace=ad9e7e43d07a42b2 ─────────────────────╯

  [1]  Chapter 10: Consistency and Consensus > Linearizability, pp. 425-427

  route=multi_hop (llm, conf 0.75) — question couples two concepts
    hop: What is a quorum in leaderless replication?
    hop: What is linearizability?
    hop: How do quorums in leaderless replication interact with linearizability?
  critic=accept faithfulness=100% citation_coverage=100%
```

<sub>Illustrative — it shows the CLI's actual rendering, with answer text and timings
standing in for your run.</sub>

---

## Why "self-verifying"

Two different things verify themselves here, and they're worth separating:

1. **The agent verifies its answers.** A critic re-reads every draft sentence against the
   chunks it cites. Unsupported claims trigger a regeneration, then a refusal, then a
   human-review escalation — in that order. The agent would rather say *"I can't support
   this"* than guess.

2. **The repo verifies itself.** `pytest` runs the entire pipeline — routing, planning,
   retrieval, generation, reflection — **offline, with no API key and no model download**,
   against a deterministic stub LLM and a hash embedder. 69 tests, ~20 seconds. CI can
   therefore gate on behaviour, not just on imports.

The stub LLM is genuinely extractive: it answers only with sentences lifted verbatim from
the retrieved chunks, and its critic re-checks those sentences term-by-term against the
cited chunk. So an offline test asserting *"the agent declines when the corpus can't
answer"* is a real assertion, not theatre.

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on POSIX
pip install -e .
cp .env.example .env                                 # then paste your ANTHROPIC_API_KEY
```

```bash
svda ingest --pdf "path/to/designing-data-intensive-applications.pdf"
```

```bash
svda ask "What is write skew?"
svda chat                                            # multi-turn, with memory
svda eval                                            # the ablation table
svda gate                                            # exits non-zero on regression
```

Everything runs without a key — the pipeline falls back to the deterministic stub and
says so in `svda info`. That's what CI uses. The ablation numbers only mean something
with a real model behind them.

---

## How the book's chapters map onto the code

| Ch. | Pattern | Where it lives | What it actually does |
|---|---|---|---|
| 14 | RAG | `ingest/`, `retrieval/`, `agent/generator.py` | ingest → chunk → embed → retrieve → cite |
| 1 | Prompt chaining | `agent/pipeline.py` | explicit stages, each with a Pydantic in/out type |
| 2 | Routing | `agent/router.py` | simple / multi-hop / out-of-scope / chit-chat |
| 6 | Planning | `agent/planner.py` | decomposes multi-hop questions into sub-questions |
| 3 | Parallelization | `retrieval/hybrid.py` | BM25 + vector + every sub-question, concurrently, then RRF |
| 4 | **Reflection** | `agent/critic.py` | per-claim faithfulness check → regenerate / decline / escalate |
| 5, 10 | Tool use / MCP | `mcp_server.py` | three tools for Claude Desktop |
| 19 | Evaluation | `eval/` | 82-item golden set, ablation runner, dashboard, CI gate |
| 12 | Exception handling | `reliability.py` | timeouts, retries, lenient JSON parsing, typed fallbacks |
| 18 | Guardrails | `agent/guardrails.py` | injection checks on queries *and* ingested documents; citation enforcement |
| 16 | Resource-aware opt. | `agent/router.py`, `llm/` | cheap model for lookups, strong model for reasoning; cost per query reported |
| 13 | Human-in-the-loop | `agent/review_queue.py` | low-confidence → queue → promoted into the golden set |
| 8 | Memory | `agent/memory.py` | resolves follow-ups like "and how does *that* interact with sharding?" |

Chapters 7 (multi-agent) and 15 (A2A) are deliberately **not** implemented. A single
corpus Q/A system gets no benefit from them, and the complexity would be unjustifiable.

---

## The headline: an ablation table

Four configurations, same corpus, same 82 golden questions, same metrics:

| Profile | What's on |
|---|---|
| `naive` | dense retrieval → stuff context → generate |
| `hybrid` | + BM25, reciprocal rank fusion, cross-encoder re-ranking |
| `agentic` | + routing and multi-hop planning with parallel sub-question retrieval |
| `full` | + reflection: faithfulness critic, regeneration, refusal |

```bash
svda eval --llm-judge          # writes runs/ablation.md and runs/dashboard.html
```

produces:

| Profile | Answer acc. | Faithfulness | Citation cov. | Multi-hop acc. | Retr. recall | Correct refusals | Mean lat. | P95 lat. | Cost/query |
|---|---|---|---|---|---|---|---|---|---|
| Naive RAG | … | … | … | … | … | … | … | … | … |
| + Hybrid + re-rank | … | … | … | … | … | … | … | … | … |
| + Routing + planning | … | … | … | … | … | … | … | … | … |
| + Reflection (full) | … | … | … | … | … | … | … | … | … |

> Run `svda eval` to populate this table from your own corpus and key. The command also
> prints the one line that matters: *"reflection moved faithfulness from X% to Y%
> (+N pts) for +M ms and +$C per query."* Numbers are deliberately not committed here —
> they're a property of your model, corpus and chunking, and a table of invented results
> would defeat the purpose of building the harness.

### The measurement decision worth defending

**Faithfulness is graded independently of the pipeline's own critic.** If the evaluator
read `answer.critic.faithfulness`, the `naive` profile (which has no critic) would score
nothing and the `full` profile would be marking its own homework — the table would show a
large, entirely fake improvement. Instead `eval/metrics.py` judges every profile the same
way: a separate critic instance (`--llm-judge`) or a deterministic extractive check,
always against the chunks the pipeline actually retrieved.

Two more choices in the same spirit:

- **Retrieval recall/precision are keyword-based, not chunk-id-based.** Pinning ground
  truth to chunk ids means every re-chunk looks like a regression. Golden items name
  terms that correct context must contain, so the metric survives a chunking change.
- **Chit-chat is excluded from faithfulness and citation coverage.** A canned "here's what
  I can help with" reply makes no factual claim; scoring it for citations would penalise
  the agent for doing the right thing.

---

## What each stage actually does

**Guardrails (input / document / output).** Deterministic pattern checks, not an LLM call —
a guardrail that needs the model fails open exactly when the model is what's under attack.
The document guardrail runs at *ingest* time, so a PDF carrying `"Ignore all previous
instructions…"` is quarantined rather than indexed. The output guardrail enforces the
rule mechanically: every factual sentence carries a resolvable citation, and any `[n]`
marker in the prose that isn't in the citation list is treated as a fabricated source.

**Router.** A cheap deterministic pass catches the obvious cases (greetings, clearly
off-topic queries) for free; only ambiguous queries pay for an LLM call. The route then
picks the model tier — lookups get Haiku, reasoning gets Opus.

**Planner → parallel retrieval.** Sub-questions are retrieved concurrently with the
original query and fused with RRF, which combines ranked lists without needing BM25 scores
and cosine similarities to be comparable. Sub-question lists are down-weighted (0.6) so a
tangent in one hop can't outvote the question actually asked. The cross-encoder runs
*once* over the merged pool rather than per sub-question, which keeps multi-hop latency
close to single-hop.

**Generator.** Emits sentences plus the context-block numbers each rests on; block numbers
are resolved back to chunk ids here, so a citation is always checkable against the index
rather than a string the model wrote. Markers are placed *inside* the sentence
(`…several machines [1].`) — put them after the period and every per-sentence check reads
the wrong sentence.

**Critic.** Proposes a verdict, but numeric thresholds decide, which is what makes the
ablation reproducible. If the critic itself is unavailable it **fails closed**: the answer
goes to human review rather than shipping as though it had been checked.

**Degradation is a feature.** If the re-ranker is down, fusion order is kept; if the strong
model times out, the cheap one answers; if a retriever dies, the other carries the query.
Each fallback is recorded and surfaced in `answer.warnings` and the trace — the system
reports being degraded rather than hiding it.

---

## CLI

| Command | Purpose |
|---|---|
| `svda ingest --pdf FILE` | chunk, guard, embed and index a PDF |
| `svda ask "…" [--profile] [--json] [--trace]` | one question |
| `svda chat` | interactive, with conversational memory |
| `svda eval [--llm-judge] [--limit N] [--baseline]` | ablation + dashboard |
| `svda gate [--tolerance]` | CI gate; non-zero exit on regression |
| `svda dashboard` | regenerate `runs/dashboard.html` from traces |
| `svda verify` | the offline self-verification suite |
| `svda review list / verdict / promote` | human-in-the-loop queue |
| `svda mcp` | run the MCP server over stdio |
| `svda info` | backend, index and golden-set status |

---

## CI gate

`svda gate` fails a PR on either of two conditions:

- **Absolute floors** — invariants that must hold on any commit (citation coverage ≥ 95%,
  faithfulness ≥ 70%, correct refusals ≥ 80%, P95 latency ≤ 60 s).
- **Relative regression** — any metric more than 3 points below the recorded baseline, so
  run-to-run noise doesn't block a PR but a genuine drop does.

```bash
svda eval --baseline     # freeze today's numbers as the baseline
svda eval && svda gate   # what CI runs on every PR
```

`.github/workflows/ci.yml` runs the offline suite on every push (no secrets needed) and
the full keyed evaluation when `ANTHROPIC_API_KEY` is available.

---

## MCP server

```json
{
  "mcpServers": {
    "self-verifying-docs": {
      "command": "C:/path/to/.venv/Scripts/svda.exe",
      "args": ["mcp"]
    }
  }
}
```

Three tools: `search_corpus` (passages with citations), `ask_corpus` (the full
fact-checked pipeline, including its refusals), `corpus_info`. The demo moment is asking
Claude Desktop something the book doesn't cover and watching it relay an honest decline
rather than a fluent invention.

---

## Layout

```
src/svda/
  config.py schemas.py tracing.py reliability.py   # settings, typed contracts, spans, fallbacks
  llm/        base stub anthropic_client tracking  # pluggable backends + usage/cost accounting
  ingest/     pdf_loader chunker pipeline          # outline-aware chunking, document guardrail
  index/      embeddings store                     # local or hash vectors, BM25, persistence
  retrieval/  retrievers fusion rerank hybrid      # parallel search, RRF, cross-encoder
  agent/      guardrails router planner generator critic memory review_queue pipeline
  eval/       golden metrics runner report ci_gate # ablation, dashboard, regression gate
  mcp_server.py cli.py
data/golden/golden_set.jsonl                       # 82 items: simple, multi-hop, out-of-scope,
tests/                                             #   chit-chat, and unanswerable
```

Citations come from the PDF's own outline, so a source is a real place in the book —
`Chapter 8: Transactions > Weak Isolation Levels, pp. 311–313` — not `chunk #412`.

---

## Known limitations

- **Faithfulness ≠ correctness.** The critic verifies that a claim is supported by the
  cited passage. A passage that's correctly cited but misread still passes.
- **Golden-set grading is keyword-based.** Cheap, deterministic and reproducible, but it
  rewards vocabulary overlap; `--llm-judge` grades faithfulness properly, answer accuracy
  stays keyword-based.
- **The 82-item golden set is hand-written**, so it reflects my reading of which questions
  matter. The review-queue → golden-set loop exists to grow it from real failures.
- **Single corpus, single document.** Nothing here handles cross-document conflicts or
  recency, both of which would change the retrieval and citation design.
