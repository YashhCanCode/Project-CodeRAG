# CodeRAG

> A production-grade **Retrieval-Augmented Generation** system for codebases and technical documentation — with hybrid retrieval, reranking, citation-enforced answers, an arXiv research-synthesis mode, full request tracing, and CI quality+cost gating.

CodeRAG is built to be *trustworthy and observable*: it combines keyword and semantic search, reranks for precision, refuses to answer when evidence is missing, **traces every request (latency, tokens, cost)**, and gates quality and cost in CI.

---

## Two modes

- **Search docs & code** — ingest code, PDFs (incl. scanned, via OCR), Markdown, HTML, notebooks, GitHub repos, or web pages, then ask questions and get answers grounded in, and citing, the exact source passages.
- **Research landscape (arXiv)** — enter a topic; CodeRAG searches arXiv, reranks for relevance, extracts each paper's problem/method/results/contribution, and synthesizes a landscape (clusters, open problems, tensions). A relevance gate refuses irrelevant topics rather than inventing a landscape.

---

## Why CodeRAG

- **Hybrid retrieval** — BM25 keyword + dense vector search, fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking** — a second-stage model re-scores candidates for precision.
- **Citation enforcement** — grounded, cited answers; refuses rather than hallucinate when evidence is thin.
- **Multi-format ingestion** — code (15+ languages), Markdown/RST/txt, PDFs (with OCR fallback for scans), HTML, notebooks, configs, GitHub repos, web pages.
- **Observability** — per-request tracing, p50/p95 latency, cost-per-request, refusal rate via a `/metrics` endpoint + JSONL traces; optional OpenTelemetry export.
- **Measured & gated** — RAGAS faithfulness eval + a CI that gates merges on deterministic tests and gates the nightly eval on faithfulness **and** a token/cost budget.
- **Pluggable LLMs** — generation and the eval judge each pick from Gemini, Groq, or local Ollama via config.
- **Minimal web UI** — Next.js + Tailwind, search-first with a Research mode.

---

## Architecture

```
Index time
  sources ──> load ──> chunk (code-aware, token-sized) ──> embed (BGE) ──> Chroma (cosine)

Query time
  question ──> ┌─ BM25 keyword search ─┐
               │                        ├─ RRF fusion ─> rerank ─> citation guard ─> LLM ─> cited answer
               └─ vector search ────────┘
               (every request is traced: stage latency, tokens, cost)

Research mode
  topic ──> arXiv search ──> rerank ──> relevance gate ──> structured extraction ──> synthesis
```

---

## Project structure

```
app/
  main.py                FastAPI entrypoint  (uvicorn app.main:app)
  state.py               shared store handle
  config/settings.yaml   all tunable knobs (models, thresholds, providers, pricing, budgets)
  routes/                health · ingest · query · research · metrics
  observability/         tracer · metrics (p50/p95, cost) · cost · otel
  services/
    ingestion/           loaders (code, pdf+OCR, html, notebook, github, web) + arxiv_loader
    chunking/            code-aware, token-based chunker
    embeddings/          BGE embedder (auto device)
    vector_store/        Chroma wrapper (cosine, idempotent)
    retrieval/           bm25 · hybrid (RRF) · reranker · pipeline
    generation/          LLM chain + citation guard
    evaluation/          RAGAS faithfulness + cost gate + golden_dataset.json
    agents/              research_landscape (arXiv synthesis agent)
  utils/                 paths/config helpers
data/                    raw · processed · vector_store (generated)
prompts/templates/       versioned prompts (rag + research)
tests/                   unit (offline, CI gate) · integration
scripts/                 try_retrieval.py (CLI)
docs/                    audit · run guide · project report
frontend/                Next.js + Tailwind UI
logs/                    per-request traces.jsonl (generated)
Dockerfile · docker-compose.yml · requirements.txt
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # GOOGLE_API_KEY (generation) + GROQ_API_KEY (eval judge)
uvicorn app.main:app --reload          # http://localhost:8000  (docs at /docs)

# CLI (retrieval-only needs no key)
python scripts/try_retrieval.py --repo https://github.com/owner/name --query "how is auth handled?"

# Frontend
cd frontend && npm install && cp .env.local.example .env.local && npm run dev   # :3000

# Docker
docker compose up --build
```

Scanned PDFs additionally need OCR: `brew install tesseract` (the libs `pymupdf pytesseract pillow` are in requirements).

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | service info |
| GET  | `/health` | liveness + whether a store is loaded |
| POST | `/ingest` | ingest a local path, GitHub repo, or web page |
| POST | `/ingest/upload` | ingest uploaded files |
| POST | `/query` | retrieve + (optional) cited answer; fully traced |
| POST | `/research` | arXiv research landscape for a topic |
| GET  | `/metrics` | aggregated p50/p95 latency, cost, tokens, refusal rate |

---

## Observability

Every `/query` is wrapped in a request trace that records **per-stage latency** (retrieval, generation), **token usage**, **estimated cost**, and the **refusal flag**. Traces are appended to `logs/traces.jsonl`, and aggregated in memory behind `GET /metrics`:

```jsonc
// GET /metrics
{
  "requests": 42,
  "latency_ms": { "p50": 820.4, "p95": 1910.0, "max": 2630.1 },
  "stage_latency_ms": { "retrieval": {...}, "generation": {...} },
  "tokens": { "total": 53210, "avg_per_request": 1267.0 },
  "cost_usd": { "total": 0.0231, "avg_per_request": 0.00055 },
  "refusal_rate": 0.07
}
```

Costs come from a pricing table in `settings.yaml` (`observability.pricing`, USD per 1M tokens). For production, enable **OpenTelemetry** export with `OTEL_TRACES=1` (console) or by setting `OTEL_EXPORTER_OTLP_ENDPOINT` — the same spans then flow to your backend (Jaeger/Grafana). It's a no-op otherwise.

---

## Evaluation & CI

```bash
python app/services/evaluation/eval.py --subset 5    # faithfulness + cost report
pytest tests/unit                                     # fast offline tests (the CI gate)
```

The eval ingests CodeRAG's own code (dogfooding), runs the real pipeline per question, and reports **faithfulness** (RAGAS), **avg tokens/request**, and **avg cost/request**. It fails if faithfulness drops below `evaluation.min_faithfulness` **or** avg tokens exceed `evaluation.max_tokens_per_request`.

CI (`.github/workflows/eval_ci.yml`) has two jobs:
- **`unit-tests`** — runs on every PR/push; the hard merge gate (free, deterministic, no keys).
- **`faithfulness-eval`** — nightly/manual only and non-blocking, so a flaky free-tier 429 never fails a PR. Needs `GOOGLE_API_KEY` + `GROQ_API_KEY` as Actions secrets.

---

## Configuration

Everything lives in `app/config/settings.yaml`: chunk size, embedding model, retrieval `top_k`, hybrid/rerank toggles, citation thresholds, **generation/judge providers** (`gemini`/`groq`/`ollama`), **pricing**, **research** gate, and **evaluation** budgets. Prompts are versioned in `prompts/templates/prompts.yaml`.

---

## Tech stack

Python · LangChain · ChromaDB · sentence-transformers (BGE embed + rerank) · rank-bm25 · FastAPI · RAGAS · OpenTelemetry · Google Gemini / Groq / Ollama · Next.js + Tailwind.

See `docs/PROJECT_REPORT.md` for the full write-up.
