# CodeRAG — End-to-End Testing Guide

Test every feature, in order. Each step says **what to run**, **what you should see**, and **what it proves**. Steps marked 🔑 need an API key; everything before them runs free.

---

## 0. Setup

```bash
cd ~/CodeRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env:
#   GOOGLE_API_KEY=...      (generation)
#   GROQ_API_KEY=...        (eval judge)
#   LANGFUSE_* (optional)
```
Optional extras:
- Scanned-PDF OCR: `brew install tesseract`
- Local generation: install Ollama + `ollama pull mistral`

> First retrieval run downloads the embedder (~130 MB) and reranker (~1.1 GB); cached after.

---

## 1. Offline unit tests (free, no keys) — the CI gate

```bash
pytest tests/unit -q
```
**Expect:** `7 passed`.
**Proves:** RRF fusion math, the code-aware BM25 tokenizer, chunker citations, and the observability math (percentiles, cost, coverage/error rate).

---

## 2. Retrieval over a GitHub repo (no key) — Phase 1 + 2

```bash
python scripts/try_retrieval.py --repo https://github.com/YashhCanCode/Local-SLM-App-with-Ollama --query "how does /extract handle retries?"
```
**Expect:** loader clones the repo → chunks → embeds → top-k cards, each with `score`, `rerank_prob`, `sim`, `bm25`, and a `<file lines X-Y>` citation.
**Proves:** ingestion (GitHub), code-aware chunking, embeddings, Chroma store, hybrid BM25+vector, RRF fusion, cross-encoder reranking, citations.

## 3. Retrieval over local files + a PDF (no key)

```bash
python scripts/try_retrieval.py --path ~/Downloads/SomeDoc.pdf --query "what is the main point?"
python scripts/try_retrieval.py --path ./docs --query "what distance metric is used?"
```
**Expect:** PDF cited per page (`p.N`); a folder ingests all supported files.
**Proves:** local file + folder ingestion, PDF (per-page) loading, page-aware citations.

## 4. Scanned PDF via OCR (no key, needs tesseract)

```bash
python scripts/try_retrieval.py --path ~/Downloads/scanned.pdf --query "..."
```
**Expect:** log shows `No text layer …; attempting OCR…` then `OCR extracted text from N page(s)`.
**Proves:** the OCR fallback for image-only PDFs.

## 5. Retrieval over a web page (no key)

```bash
python scripts/try_retrieval.py --url https://en.wikipedia.org/wiki/Retrieval-augmented_generation --query "what is RAG?"
```
**Proves:** web-page ingestion (server-rendered HTML).

---

## 6. 🔑 Generation + citation guard — Phase 1 + 2

```bash
python scripts/try_retrieval.py --repo https://github.com/YashhCanCode/Local-SLM-App-with-Ollama --query "how does /extract handle retries?" --answer
python scripts/try_retrieval.py --no-ingest --query "what is the airspeed velocity of a swallow?" --answer
```
**Expect:** first → grounded ANSWER with CITATIONS; second → **REFUSED** ("Insufficient evidence").
**Proves:** LLM generation grounded in retrieved chunks, and the citation guard refusing rather than hallucinating.

---

## 7. The API server — all endpoints

```bash
uvicorn app.main:app --reload     # http://localhost:8000  · docs at /docs
```
In another terminal:

```bash
# service info + health
curl localhost:8000/ ; echo
curl localhost:8000/health ; echo

# ingest (github / web)  — or reuse an existing chroma_db and skip
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"source_type":"github","path":"https://github.com/YashhCanCode/Local-SLM-App-with-Ollama"}'

# upload a file (multipart)
curl -F "files=@/path/to/file.pdf" localhost:8000/ingest/upload
# upload an UNSUPPORTED file -> helpful 400 listing supported types
curl -F "files=@/path/to/data.csv" localhost:8000/ingest/upload

# retrieval-only (no LLM, no key burned)
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"how does /extract handle retries?","generate":false}'

# 🔑 full cited answer
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"how does /extract handle retries?"}'
```
**Proves:** every route, the file-upload path (incl. the helpful error), and retrieval-only vs. generated answers.

---

## 8. 🔑 Research landscape (arXiv) — the agent

```bash
# a real research area -> a landscape
curl -X POST localhost:8000/research -H 'content-type: application/json' \
  -d '{"topic":"retrieval augmented generation evaluation","max_papers":6}'

# a tool name / question -> relevance gate REFUSES
curl -X POST localhost:8000/research -H 'content-type: application/json' \
  -d '{"topic":"what is RAGAS?"}'
```
**Expect:** first → `papers` (real arXiv, with problem/method/results/contribution + cluster) and `synthesis` (overview, clusters, open_problems, tensions). Second → `404` with "No sufficiently relevant arXiv papers…".
**Proves:** arXiv search, reranking, structured extraction, cross-paper synthesis, and the relevance gate.

---

## 9. Observability — Project 3

```bash
# after running a few /query requests:
curl localhost:8000/metrics ; echo
tail -n 2 logs/traces.jsonl
```
**Expect (`/metrics`):** `requests`, `latency_ms` (p50/p95/max), `stage_latency_ms` (retrieval, generation), `tokens`, `cost_usd`, `refusal_rate`, `citation_coverage`, `error_rate`.
**Expect (trace line):** the retrieved chunks (+scores), the **exact prompt**, the **answer**, tokens, and cost per request.
**Proves:** full request tracing, p50/p95 latency, cost-per-request, citation coverage, failure rate.

Optional integrations:
```bash
OTEL_TRACES=1 uvicorn app.main:app --reload      # prints OpenTelemetry spans to console
# Langfuse: set LANGFUSE_PUBLIC_KEY/SECRET_KEY in .env, run queries -> traces + dashboard in the Langfuse UI
```

---

## 10. 🔑 Evaluation + regression gate — Phase 3 + Project 3

```bash
python app/services/evaluation/eval.py --subset 3
```
**Expect:** per-question lines, then a block with `Mean faithfulness`, `Scored x/3`, `Avg tokens/req` (vs budget), `Avg cost/req`, and `PASS ✅` / `FAIL ❌`.
**Proves:** RAGAS faithfulness scoring (Gemini generates, Groq judges), plus the **cost/token budget gate** alongside faithfulness.

> Quota note: generation uses Gemini's free tier (low daily cap). If it 429s, switch `generation.provider` to `ollama` or `groq` in `settings.yaml`, or wait for reset.

---

## 11. The web UI

```bash
cd frontend && npm install && cp .env.local.example .env.local && npm run dev   # :3000
```
- **Search docs & code:** pick a tab (Upload / GitHub / Web) → Ingest → ask a question → answer + source cards → click a card → full source **slides in from the left** with syntax highlighting.
- **Research landscape:** enter a topic → overview, open problems, tensions, papers grouped by cluster → click a paper → slide-in detail with an arXiv link.
**Proves:** the full UI over both modes (backend must be running on :8000).

---

## 12. CI (on GitHub)

- Add repo **Actions secrets**: `GOOGLE_API_KEY`, `GROQ_API_KEY`.
- Open a PR → the **`unit-tests`** job runs and gates the merge (free, deterministic).
- Trigger **`faithfulness-eval`** via Actions → "Run workflow" (or wait for the nightly) → faithfulness + cost report (non-blocking).
**Proves:** CI regression gating wired in, with the LLM eval kept off the PR path to avoid quota flakiness.

---

### Fastest smoke test (free, ~2 min)
```bash
pytest tests/unit -q
python scripts/try_retrieval.py --repo https://github.com/YashhCanCode/Local-SLM-App-with-Ollama --query "how is retry handled?"
```
If both look good, the core pipeline (ingest → chunk → embed → hybrid retrieve → rerank → cite) is working without spending a single API call.
