# CodeRAG — Project Report

**A production-style Retrieval-Augmented Generation system for codebases and technical documentation — with hybrid retrieval, reranking, citation-enforced answers, an arXiv research-synthesis mode, full request observability, and CI quality + cost gating.**

CodeRAG ingests code and technical documents (PDFs incl. scanned, Markdown, HTML, notebooks, web pages, GitHub repos), indexes them, and answers questions grounded in — and citing — the exact source passages. It was built in phases (working pipeline → production retrieval → evaluation → research-synthesis agent → observability), and reorganized into a clean service-oriented layout with a web UI.

---

## At a glance

```
Index time:   load sources → chunk → embed → store in Chroma (cosine)
Query time:   question → hybrid (BM25 + vector) → RRF fusion → cross-encoder rerank
                       → citation guard → grounded answer        (every request traced)
Research mode: topic → arXiv search → rerank → relevance gate → extraction → synthesis
```

**Stack:** Python · LangChain · ChromaDB · BAAI/bge-small (embed) · BAAI/bge-reranker-base (rerank) · rank-bm25 · FastAPI · RAGAS · OpenTelemetry · Next.js + Tailwind.

**LLM providers (pluggable via config):**
- **Generation:** Google **Gemini 2.5 Flash** (default); also Groq or local Ollama.
- **Eval judge:** **Groq** `openai/gpt-oss-20b` — separate from generation so judging is independent and free.

---

## Phase 1 — The pipeline

Ingestion of code (15+ languages), Markdown/RST/text, **PDFs** (per page, **OCR fallback** for scans), **HTML**, **Jupyter notebooks**, config files, plus **GitHub repos** and **web pages**. Code-aware chunking (~500–800 tokens, splitting on function/class boundaries), cosine vector storage in ChromaDB with idempotent re-ingestion, and retrieval with accurate **file/line/page citations**. An initial review caught and fixed real bugs — most notably an **inverted relevance score** (distance treated as similarity), token-vs-character chunk sizing, and citation line-range errors on duplicated code.

## Phase 2 — Production-quality retrieval

**Hybrid retrieval** (BM25 keyword + dense vector, fused with **Reciprocal Rank Fusion**), **cross-encoder reranking** (`bge-reranker-base`), **citation enforcement** (an architectural guard that refuses before calling the LLM when evidence is insufficient), and **configuration management** (all knobs + versioned prompts in config files).

## Phase 3 — Evaluation & CI

A **golden dataset**, **RAGAS faithfulness** evaluation (generation on Gemini, judging on Groq), and **CI regression gating**. The harness scored **0.833 faithfulness (PASS)** on a verified subset. CI was later split so that deterministic unit tests gate every PR while the LLM eval runs nightly/non-blocking (free-tier LLM evals are too flaky to hard-gate PRs).

## Research landscape (arXiv agent)

A second mode that turns a topic into a structured map of the literature: **arXiv search → cross-encoder rerank → relevance gate → batched structured extraction (problem/method/results/contribution) → cross-paper synthesis (clusters, open problems, tensions)**. Two LLM calls per topic (batched) to stay within free-tier limits; runs on abstracts (fast, free). A **relevance gate** refuses irrelevant topics (e.g. a tool name or a question) instead of synthesizing a landscape from unrelated papers — the research-mode analogue of the citation guard. Lives in `app/services/agents/`.

## Project 3 — Monitoring & observability

Production-ops instrumentation across the system:

- **Per-request tracing** — every `/query` records per-stage latency (retrieval, generation), token usage, estimated cost, and the refusal flag; traces are written to `logs/traces.jsonl`.
- **Metrics endpoint** — `GET /metrics` aggregates **p50/p95 latency** (overall and per stage), total/avg **cost**, total/avg **tokens**, and **refusal rate** from an in-memory ring buffer.
- **Cost model** — token→USD via a pricing table in `settings.yaml`; generation reports real token usage from the LLM response.
- **OpenTelemetry** — optional, opt-in via `OTEL_TRACES=1` or an OTLP endpoint; the same spans flow to a backend (Jaeger/Grafana) in production, no-op otherwise.
- **CI cost gate** — the eval now also reports avg tokens/cost per request and **fails on a token budget** alongside faithfulness (tokens are machine-stable, so this is not flaky).

## Web frontend (Next.js + Tailwind)

A minimal, search-first UI with a **Search / Research** mode toggle. Search mode: ingest (upload / GitHub / web) → answer + source cards, click a card for a left **slide-in panel** with the syntax-highlighted source. Research mode: topic → landscape overview, open-problems/tensions, papers grouped by cluster, slide-in detail with an arXiv link. Source cards always render even if answer generation is rate-limited, so search works regardless of LLM quota.

---

## Project structure (service-oriented `app/` layout)

```
app/
  main.py · state.py
  config/settings.yaml
  routes/        health · ingest · query · research · metrics
  observability/ tracer · metrics · cost · otel
  services/
    ingestion/ (+ arxiv_loader) · chunking · embeddings · vector_store
    retrieval/ · generation/ · evaluation/ · agents/ (research_landscape)
  utils/
data/ · prompts/templates/ · tests/ · scripts/ · docs/ · logs/ · frontend/
Dockerfile · docker-compose.yml
```

---

## Current status

| Area | Status |
|---|---|
| Phase 1 pipeline | Complete, verified on real repos + PDFs (OCR for scans) |
| Phase 2 hybrid + rerank + citations | Complete, verified |
| Phase 3 eval + CI gating | Complete; 0.833 faithfulness PASS on subset |
| arXiv research landscape | Complete; relevance-gated, verified |
| Project 3 observability | Complete; `/metrics`, tracing, cost, OTel hooks, CI cost gate |
| `app/` restructure | Complete; compiles, API + pipeline + tests verified |
| Web frontend | Complete, builds + type-checks clean |

**Verified in the latest check:** all modules compile; config loads (generation `gemini-2.5-flash`, judge `groq`); all seven API routes respond; unit tests pass (incl. observability math); a traced `/query` populates `/metrics` correctly; the cost gate fires on budget; OTel is off by default.

**Open (operational):** push to GitHub + add CI secrets (`GOOGLE_API_KEY`, `GROQ_API_KEY`); grow the golden dataset toward 50–200 pairs; optionally add the v3 interactive research graph and full-PDF deep-read.

---

*A working pipeline, production-grade retrieval, a regression-and-cost-gated evaluation loop, an arXiv research-synthesis agent, and full request observability — in a clean service layout with a web interface.*
