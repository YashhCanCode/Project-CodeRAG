# CodeRAG — Phase 1 Audit

**Date:** 2026-06-19
**Scope:** Review-only (no code changed). Findings backed by running the pipeline modules in a Linux sandbox.
**Phase 1 definition used:** ingest -> chunk -> vector store -> retrieve top-k with citations. Generation (Gemini) noted but not gated on.

## Verdict

The scaffold is well-organized and the pieces are all present, but **Phase 1 is not yet correct**. There is one **critical, confirmed bug that inverts retrieval quality** (the citation gate keeps irrelevant chunks and drops exact matches), plus a chunk-size deviation from the spec and a citation line-range inaccuracy. None require an architecture change — they're focused fixes. Fix the four CRITICAL/HIGH items and Phase 1 is solid.

> Note on verification: the sandbox proxy blocks HuggingFace/tiktoken downloads, so the real `BAAI/bge-small-en-v1.5` embedder was **not** run live. Every confirmed finding below is **embedder-independent** (it's about Chroma scoring, the chunker, and the gate logic), so the conclusions hold regardless of which embedding model is used. Token counts use the standard ~4-chars/token heuristic.

---

## [CRITICAL]

### 1. The citation/confidence gate is inverted — it drops the best matches and keeps irrelevant ones
**Where:** `generation/chain.py` -> `trusted = [c for c in chunks if c["score"] >= CITATION_SCORE_THRESHOLD]` (threshold `0.3`), and the docstrings in `retrieval/vector_store.py` / `retrieval/pipeline.py` that call the score a "cosine similarity — higher is better."

**Root cause:** `Chroma.similarity_search_with_score()` returns a **distance** (lower = more similar), and the collection is created with **no `hnsw:space` set, so it defaults to L2**, not cosine. The code treats the number as a similarity where higher is better.

**Confirmed live** (toy embedder, default Chroma collection):

```
Query: 'auth login token'
  score=0.0000  auth.py 1-3   def login(): validate auth token   <- perfect match
  score=2.0000  db.py 1-3     def connect_db(): ...
  score=2.0000  pay.py 1-3    def charge(): ...
collection space metadata: None        # => default L2 distance

generation/chain.py gate (score >= 0.3) keeps: ['db.py 1-3', 'pay.py 1-3']
  => the exact match (distance 0.0) is DROPPED; the two irrelevant chunks PASS.
```

**Impact:** Worst-possible retrieval behavior — the system answers from the least relevant chunks and refuses (or mis-cites) when it actually has the right code. This defeats the entire point of Phase 1.

**Fix direction (your call to implement):** decide on one score convention and make it consistent end-to-end. Either (a) create the collection with `collection_metadata={"hnsw:space": "cosine"}` and convert to similarity (`similarity = 1 - distance`) before comparing, or (b) keep distance and flip the gate to `distance <= threshold` with a distance-appropriate cutoff. Pipeline ordering (`raw_results[:top_k]`) is fine because Chroma already returns results sorted best-first; only the *interpretation* of the number is wrong.

---

## [HIGH]

### 2. Chunks are ~4x smaller than the spec (chars vs. tokens)
**Where:** `config/settings.yaml` (`chunk_size: 600`, `chunk_overlap: 100`) + `ingestion/chunker.py` (`RecursiveCharacterTextSplitter(..., length_function=len)`).

`length_function=len` counts **characters**, so `chunk_size: 600` ~= **150 tokens**, and overlap `100` ~= 25 tokens. The Phase 1 spec asks for **500-800 tokens with ~100-token overlap**.

**Confirmed:** a 137-char sample file produced one chunk of ~34 tokens; `600 chars ~= 150 tokens`. To hit 500-800 tokens you'd need roughly **2,000-3,200 characters** (or a token-based `length_function`).

**Impact:** Over-fragmented context. Functions/classes get split more than intended, retrieval returns less surrounding context per hit, and answers lose the boundary context the overlap was meant to preserve.

**Fix direction:** either pass a token-based `length_function` (e.g. a tiktoken/HF tokenizer count) and keep 500-800, or convert the spec to characters (~2,400 / ~400) and document that the units are characters.

### 3. Citation line ranges are wrong for repeated/duplicate code
**Where:** `ingestion/chunker.py` -> `_estimate_line_range()` uses `full_text.find(chunk_text)`, which returns the **first** occurrence and **ignores `chunk_index`**.

**Confirmed:** a file with two identical `x = 1 / return x` bodies (the real one at lines 8-9) resolved to **lines 4-5** — the first copy. Overlapping chunks and any repeated boilerplate (imports, getters, similar handlers) will mis-cite.

**Impact:** Citations are the headline feature of Phase 1, and they can silently point to the wrong location. Erodes the core trust guarantee.

**Fix direction:** track a running character offset per document while splitting (search from the previous chunk's end, or have the splitter emit start indices) instead of a global `find()`.

---

## [MEDIUM] (workflow / correctness)

### 4. Re-ingesting the same repo duplicates vectors (no reset/dedup)
**Where:** `retrieval/vector_store.py` -> `build_vector_store()` calls `Chroma.from_documents(...)`, and `api.py /ingest` calls it every time.

**Confirmed:** ingesting identical content twice took the collection from **1 -> 2** vectors. Repeated ingests inflate the index with duplicates, skewing retrieval and counts.

**Fix direction:** reset/delete the collection before a rebuild, or use stable IDs (e.g. `source + chunk_index` hash) with `upsert` so re-ingest is idempotent.

### 5. `mps` device is hardcoded
**Where:** `retrieval/embedder.py` -> `model_kwargs={"device": "mps"}`. The comment says it "falls back to CPU automatically," but sentence-transformers does **not** silently fall back — on any non-Apple-Silicon machine (e.g. the empty `eval_ci.yml` CI runner, a Linux box) this raises. Fine on your Mac today; a portability/CI landmine.

**Fix direction:** detect the device (`cuda`/`mps`/`cpu`) or make it a config value.

### 6. Everything assumes the current working directory is the project root
**Where:** `chunker.py`, `embedder.py`, `vector_store.py`, `pipeline.py` all do `open("config/settings.yaml")`, and `persist_directory: "./chroma_db"` is relative. Run the app from any other directory and config loading and the DB path break.

**Fix direction:** resolve paths relative to the file (`Path(__file__).parent`) or a project-root constant.

### 7. The in-memory `_store` handle doesn't survive `reload`/multiple workers
**Where:** `api.py` runs `uvicorn.run(..., reload=True)` and keeps `_store` as a module global. With reload or >1 worker, `/query` will often hit the lazy `load_vector_store()` path (acceptable) but the `/ingest`-populated handle isn't shared across workers. Minor for single-process local dev; worth knowing before you scale.

---

## [LOW] / polish

- **Deprecated import:** `langchain_community.embeddings.HuggingFaceBgeEmbeddings` is deprecated in favor of `langchain_huggingface.HuggingFaceEmbeddings`. Works today, will warn/break later.
- **Unpinned `requirements.txt`:** no version pins => non-reproducible installs. Pin versions before Phase 2.
- **Code separators are Python/JS-centric:** `"\nclass "`, `"\ndef "` won't match Go/Rust/Java/C. Consider `RecursiveCharacterTextSplitter.from_language(...)` per language (you already store `language` in metadata).
- **Citation format mismatch:** chunker emits `"source lines X-Y"` (en-dash) while `prompts.yaml` instructs the model to output `[file_path, lines X-Y]`. Cosmetic, but the model is told to produce a format that doesn't match the metadata it's handed.
- **`vector_search_k: 20` is fetched then sliced to `top_k` with no reranking** in `pipeline.py`. Correct for Phase 1 (results are pre-sorted), just wasted fetch until the Phase 2 reranker lands.
- **Empty Phase 2/3 stubs** (`bm25.py`, `hybrid.py`, `reranker.py`, `citation_guard.py`, `evaluation/eval.py`, `golden_dataset.json`, `eval_ci.yml`) — expected, just flagging they're empty.

---

## Suggested order to "perfect the workflow" before Phase 2

1. **#1 score/gate** — correctness blocker; nothing downstream is trustworthy until this is fixed.
2. **#3 line ranges** — citations must be right; it's the Phase 1 promise.
3. **#2 chunk size** — bring chunks to the spec'd token range.
4. **#4 re-ingest dedup** — makes the ingest workflow repeatable.
5. **#5 device + #6 paths** — portability so CI / a fresh clone runs.
6. Polish items as time allows.

Once #1-#4 are done, a quick end-to-end check (ingest a small repo -> ask a question whose answer lives in one known file -> confirm the top chunk and its citation point at that file) is enough to call Phase 1 complete. A handful of those Q->expected-file pairs is also the natural seed for the Phase 3 `golden_dataset.json`.
