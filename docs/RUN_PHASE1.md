# Phase 1 — How to test in VS Code

## What's required
- **Embedding model: yes.** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (imported in `retrieval/embedder.py`). First run auto-downloads ~130MB and caches; device is auto-detected (mps on your Mac).
- **LLM (Gemini): NOT required for retrieval.** Only the `/query` endpoint's answer-generation step uses it. The `try_retrieval.py` script and the retrieval pipeline need no API key.

## Setup
```bash
cd /path/to/CodeRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Fastest check — retrieval only, no LLM
```bash
# ingest a codebase and run one query
python try_retrieval.py --path ./some_repo --query "how is auth handled?"

# or interactive
python try_retrieval.py --path ./some_repo

# reuse the store you already built
python try_retrieval.py --query "where is the retry logic?" --no-ingest
```
You should see top-k chunks with `score` (cosine similarity, higher = better) and a `<file lines X-Y>` citation. Sanity test: ask something whose answer lives in one known file and confirm the top hit cites that file.

## Full API (retrieval + generation)
Generation needs a key. Create `.env` in the project root:
```
GOOGLE_API_KEY=your_key_here
```
Then:
```bash
uvicorn api:app --reload
# ingest
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"source_type":"local","path":"./some_repo","repo_name":"demo"}'
# query
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"how is auth handled?"}'
```


## Supported sources (codebases + technical docs)
- **Code:** .py .js .ts .tsx .jsx .go .rs .java .cpp .c .h .rb .php .cs .swift .kt
- **Docs:** .md .mdx .rst .txt, **.pdf** (per-page, cited as `file p.N`), **.html/.htm**, **.ipynb** (cells flattened)
- **Config:** .json .yaml/.yml .toml .ini .env
- **Web pages:** ingest a URL directly

```bash
# a PDF/HTML/notebook sitting in a folder is picked up automatically
python try_retrieval.py --path ./docs --query "what database is used?"

# ingest a web page
python try_retrieval.py --url https://example.com/docs/page --query "how do I configure rate limits?"
```
API: `POST /ingest` now accepts `"source_type": "web"` (with `path` = the URL) in addition to `local` and `github`.

## What changed in this pass (see PHASE1_AUDIT.md for the why)
- **#1** Chroma now uses cosine space; `pipeline.retrieve()` returns a **similarity** (higher = better). The citation gate in `generation/chain.py` is no longer inverted.
- **#2** Chunking is **token-based** (tiktoken, 500–800-token range) with a char fallback if offline.
- **#3** Citation line ranges use a forward-moving cursor, so duplicate code maps to the correct copy.
- **#4** Re-ingest is **idempotent** (collection reset + stable `chunk_id`s) — no duplicate vectors.
- **#5/#6** Embedding device auto-detects (cuda/mps/cpu); config + DB paths resolve to the project root via `paths.py`, so you can run from any directory.

Note: tiktoken and the BGE model couldn't be exercised in the build sandbox (no network), but every fix above was verified end-to-end there with a stand-in embedder and small chunks. The token splitter and BGE will work on your machine where downloads are allowed.
