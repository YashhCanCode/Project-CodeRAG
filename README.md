# CodeRAG

> A production-grade **Retrieval-Augmented Generation** system for codebases and technical documentation. Point it at code, PDFs, Markdown, web pages, or notebooks — then ask questions and get answers grounded in, and citing, the exact source passages.

CodeRAG is built to be *trustworthy*: it combines keyword and semantic search, reranks for precision, refuses to answer when the evidence isn't there, and ships with an automated evaluation loop that gates quality on every change.

---

## Why CodeRAG

- **Hybrid retrieval** — BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion, so it catches both exact identifiers (`create_agent`, `.env`, `thermal`) and meaning.
- **Cross-encoder reranking** — a second-stage model re-scores candidates for sharp, precise ordering.
- **Citation enforcement** — answers are grounded in retrieved chunks with file/line/page citations, and the system *refuses* rather than hallucinate when evidence is insufficient.
- **Multi-format ingestion** — code (15+ languages), Markdown/RST/txt, PDFs (per page), HTML, Jupyter notebooks, config files, GitHub repos, and web pages.
- **Measured, not vibes** — RAGAS faithfulness evaluation with a golden dataset and a CI regression gate.
- **Pluggable LLMs** — generation and the eval judge each pick from Gemini, Groq, or local Ollama via config.
- **Clean web UI** — a minimal Next.js + Tailwind search interface.

---

## Architecture

```
Index time
  sources ──> load ──> chunk (code-aware, token-sized) ──> embed (BGE) ──> Chroma (cosine)

Query time
  question ──> ┌─ BM25 keyword search ─┐
               │                        ├─ RRF fusion ─> cross-encoder rerank ─> citation guard ─> LLM ─> cited answer
               └─ vector search ────────┘
```

The retrieval pipeline returns ranked chunks with citations; generation is optional (retrieval works key-free).

---

## Project structure

```
app/
  main.py                FastAPI entrypoint  (uvicorn app.main:app)
  state.py               shared in-memory store handle
  config/settings.yaml   all tunable knobs (models, thresholds, providers)
  routes/                ingest · query · health route modules
  services/
    ingestion/           loaders: code, pdf, html, notebook, github, web
    chunking/            code-aware, token-based chunker
    embeddings/          BAAI/bge-small-en-v1.5 embedder (auto device)
    vector_store/        Chroma wrapper (cosine, idempotent re-ingest)
    retrieval/           bm25 · hybrid (RRF) · reranker · pipeline
    generation/          LLM chain + citation guard
    evaluation/          RAGAS faithfulness eval + golden_dataset.json
    agents/              reserved for future agentic features
  utils/                 paths/config helpers
data/                    raw · processed · vector_store (generated, gitignored)
prompts/templates/       versioned prompts
tests/                   unit · integration
scripts/                 try_retrieval.py (CLI)
docs/                    audit, run guide, project report
frontend/                Next.js + Tailwind search UI
Dockerfile · docker-compose.yml · requirements.txt
```

---

## Quickstart

### 1. Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # add GOOGLE_API_KEY (generation) and GROQ_API_KEY (eval judge)
uvicorn app.main:app --reload
```
API at `http://localhost:8000` · interactive docs at `/docs`.

> First query downloads the embedder (~130MB) and reranker (~1.1GB), cached afterward.

### 2. CLI (no server, no API key for retrieval)

```bash
python scripts/try_retrieval.py --repo https://github.com/owner/name --query "how is auth handled?"
python scripts/try_retrieval.py --path ./docs --query "what database is used?"
python scripts/try_retrieval.py --url https://docs.example.com/page
# add --answer to also generate a cited answer (needs an LLM key)
```

### 3. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local      # points at http://localhost:8000
npm run dev                            # http://localhost:3000
```

### 4. Docker

```bash
docker compose up --build
```

---

## Configuration

Everything lives in `app/config/settings.yaml` — no code edits needed:

| Setting | What it controls |
|---|---|
| `chunking.chunk_size / chunk_overlap` | 600 / 100 tokens (code-aware splitting) |
| `embedding.model` | `BAAI/bge-small-en-v1.5` |
| `retrieval.top_k / vector_search_k / bm25_k` | how many chunks to fetch / return |
| `hybrid.enabled / rrf_k` | toggle hybrid fusion; RRF constant (60) |
| `reranker.enabled / model` | cross-encoder reranking (`BAAI/bge-reranker-base`) |
| `citation.min_*` | evidence thresholds for the refuse-or-answer gate |
| `generation.provider / model` | `gemini` · `groq` · `ollama` |
| `evaluation.judge_provider / model` | `groq` · `gemini` |

Prompts are versioned in `prompts/templates/prompts.yaml`.

---

## Evaluation

```bash
python app/services/evaluation/eval.py --subset 5    # quick
python app/services/evaluation/eval.py               # full golden set
pytest tests/unit                                     # fast offline unit tests
```

The eval ingests CodeRAG's own code (dogfooding), runs the real pipeline per question, and scores **faithfulness** with RAGAS — are the answer's claims supported by the retrieved evidence? It **exits non-zero if faithfulness drops below the threshold**, so the GitHub Actions workflow (`.github/workflows/eval_ci.yml`) blocks regressions on pull requests.

---

## Tech stack

Python · LangChain · ChromaDB · sentence-transformers (BGE embed + rerank) · rank-bm25 · FastAPI · RAGAS · Google Gemini / Groq / Ollama · Next.js + Tailwind.

---

## Built in three phases

1. **Pipeline** — ingestion, code-aware chunking, vector storage, cited retrieval.
2. **Production retrieval** — hybrid BM25+vector, reranking, citation enforcement, config management.
3. **Evaluation & CI** — golden dataset, RAGAS faithfulness, regression-gated CI.

See `docs/PROJECT_REPORT.md` for the full write-up.
