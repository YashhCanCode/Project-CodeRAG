# CodeRAG — Project Report

**A production-style Retrieval-Augmented Generation system for codebases and technical documentation.**

CodeRAG ingests code repositories and technical documents (PDFs, Markdown, web pages, notebooks), indexes them, and answers questions with responses that are grounded in — and cite — the exact source passages. It was built in three phases (working pipeline → production-grade retrieval → measurable, regression-gated evaluation), reorganized into a clean service-oriented layout, and given a minimal web interface.

---

## At a glance

```
Index time:   load sources → chunk → embed → store in Chroma (cosine)
Query time:   question → hybrid search (BM25 + vector) → RRF fusion
                       → cross-encoder rerank → citation guard → grounded answer
```

**Stack:** Python · LangChain · ChromaDB · BAAI/bge-small (embeddings) · BAAI/bge-reranker-base (reranking) · rank-bm25 · FastAPI · RAGAS (evaluation) · Next.js + Tailwind (frontend).

**LLM providers (pluggable via config):**
- **Generation:** Google **Gemini 2.5 Flash** (default) — also supports Groq and local Ollama.
- **Evaluation judge:** **Groq** `openai/gpt-oss-20b` — kept separate from generation so faithfulness judging is independent and free.

---

## Phase 1 — The pipeline

Established the core ingest-to-answer path:

- **Ingestion** of code (15+ languages), Markdown/RST/text, **PDFs** (per page), **HTML**, **Jupyter notebooks**, config files, plus **GitHub repos** (auto-cloned) and **web pages**.
- **Code-aware chunking** at ~500–800 tokens with overlap, splitting on function/class boundaries so logical units stay intact.
- **Vector storage** in ChromaDB using cosine similarity, with idempotent re-ingestion (no duplicate vectors).
- **Retrieval** returning the top-k passages with accurate **file + line/page citations**.

A review at the start of this phase caught and fixed several real bugs, most notably an **inverted relevance score** (the system had been treating distance as similarity, ranking the *least* relevant chunks first), token-vs-character chunk sizing, and citation line-range errors on duplicated code.

## Phase 2 — Production-quality retrieval

Raised precision from "demo" to "professional":

- **Hybrid retrieval** — BM25 keyword search (exact terms like identifiers, `thermal`, `.env`) combined with dense vector search (meaning), fused via **Reciprocal Rank Fusion**.
- **Cross-encoder reranking** — re-scores the fused candidates with `bge-reranker-base` for a sharp final ordering.
- **Citation enforcement** — an architectural guard that refuses to answer when retrieval lacks sufficient evidence, *before* calling the LLM, preventing hallucination.
- **Configuration management** — all knobs (models, thresholds, fusion constants, providers) and versioned prompts live in config files, not code.

## Phase 3 — Evaluation & CI

Made the system measurable and shippable:

- **Golden dataset** — hand-verified question/answer pairs (seeded at 15, grows to 50–200) benchmarking the system against its own codebase.
- **Offline evaluation with RAGAS** — measures **faithfulness**: are the claims in each answer supported by the retrieved evidence? Generation runs on Gemini; judging runs on Groq.
- **CI/CD with regression gating** — a GitHub Actions workflow runs the eval on pull requests and **fails the build if faithfulness drops below 0.70**, blocking quality regressions.

**Result:** the harness runs end-to-end and scored **0.833 faithfulness (PASS)** on a verified subset.

## Web frontend (Next.js + Tailwind)

A minimal, search-first interface on the FastAPI backend:

- **Ingest panel** — upload PDFs/files, paste a GitHub repo URL, or a web page.
- **Search-first UX** — a search bar returns a synthesized **answer** plus a grid of **source cards** (one per retrieved chunk).
- **Click-through detail** — each card opens a modal with the **full source passage, syntax-highlighted**, and its citation.
- **Resilient by design** — source cards always render even if answer generation is rate-limited, so search works regardless of LLM quota.

---

## Project structure (service-oriented `app/` layout)

```
app/
  main.py                FastAPI entrypoint (uvicorn app.main:app)
  state.py               shared store handle
  config/settings.yaml   tunable knobs + provider selection
  routes/                ingest · query · health
  services/
    ingestion/ chunking/ embeddings/ vector_store/
    retrieval/ generation/ evaluation/ agents/
  utils/                 paths/config helpers
data/                    raw · processed · vector_store (generated)
prompts/templates/       versioned prompts
tests/                   unit · integration
scripts/                 try_retrieval.py (CLI)
docs/                    audit · run guide · this report
frontend/                Next.js + Tailwind UI
Dockerfile · docker-compose.yml
```

---

## Current status

| Area | Status |
|---|---|
| Phase 1 pipeline | Complete, verified on real repos + PDFs |
| Phase 2 hybrid + rerank + citations | Complete, verified |
| Phase 3 eval harness + CI workflow | Complete; 0.833 PASS on subset |
| Web frontend | Complete, builds + type-checks clean |
| `app/` restructure | Complete; compiles, API + pipeline + tests verified |

**Verified in the final check:** all modules compile; config loads (generation = `gemini-2.5-flash`, judge = `groq`); all five API routes respond; unit tests pass; the ingest → chunk → store → hybrid-retrieve pipeline runs end-to-end; the frontend type-checks.

**Open (operational, not build work):**
- One clean full eval run for a baseline (bounded by free-tier LLM limits; trivially cheap on a paid tier).
- Push to GitHub + add CI secrets (`GOOGLE_API_KEY`, `GROQ_API_KEY`).
- Grow the golden dataset toward 50–200 pairs over time.

---

*Three phases delivered — a working pipeline, production-grade retrieval, and a regression-gated evaluation loop — reorganized into a clean service layout with a web interface for demos.*
