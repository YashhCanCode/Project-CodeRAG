"""
app/services/evaluation/eval.py

Offline evaluation for CodeRAG using RAGAS.

What it measures
----------------
FAITHFULNESS (the gating metric): are the claims in the generated answer
actually supported by the retrieved chunks, or hallucinated? RAGAS uses an
LLM judge (here: Gemini, reusing your GOOGLE_API_KEY) to break each answer
into claims and check each against the retrieved context. 1.0 = every claim
supported; lower = some claims unsupported.

How it runs
-----------
1. Ingest the eval corpus (CodeRAG's OWN repo — dogfooding) into an isolated
   Chroma collection so your main store is untouched.
2. For each golden question: run the real pipeline (hybrid -> rerank ->
   citation guard -> Gemini) and collect (question, contexts, answer, reference).
3. Score with RAGAS faithfulness.
4. Regression gate: exit non-zero if mean faithfulness < threshold, so CI fails.

Usage
-----
    python app/services/evaluation/eval.py                # full golden set
    python app/services/evaluation/eval.py --subset 15    # fast PR smoke subset
    python app/services/evaluation/eval.py --min-faithfulness 0.75   # override gate

Requires GOOGLE_API_KEY (judge + generation). First run downloads the
embedder + reranker.
"""

import os
import sys
import json
import time
import argparse

# make project root importable when run as a script
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv
load_dotenv()

from app.utils.paths import PROJECT_ROOT, load_settings
from app.services.ingestion.loaders import load_local_directory
from app.services.chunking.chunker import chunk_documents
from app.services.vector_store.store import build_vector_store
from app.services.retrieval.pipeline import retrieve
from app.services.generation.chain import generate_answer

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dataset.json")


def load_golden(subset: int = None):
    with open(GOLDEN_PATH) as f:
        pairs = json.load(f)["pairs"]
    return pairs[:subset] if subset else pairs


def build_eval_store():
    """Ingest CodeRAG's own repo into an isolated collection."""
    s = load_settings()["evaluation"]
    print(f"[eval] Ingesting corpus: {PROJECT_ROOT}")
    docs = load_local_directory(str(PROJECT_ROOT), repo_name="coderag")
    chunks = chunk_documents(docs)
    return build_vector_store(
        chunks,
        persist_dir=s["persist_directory"],
        collection=s["collection_name"],
    )


def collect_samples(store, pairs, sleep_s=0.0, ctx_chunks=3, ctx_chars=900):
    """Run the real pipeline for each question and build RAGAS records.

    Contexts sent to the judge are trimmed (count + length) so each faithfulness
    request stays under the judge's tokens-per-minute limit.
    """
    records, refusals = [], 0
    for i, p in enumerate(pairs):
        if i and sleep_s:
            time.sleep(sleep_s)   # throttle to respect free-tier rate limits
        q = p["question"]
        chunks = retrieve(store, q)
        result = generate_answer(q, chunks)
        if result["refused"]:
            refusals += 1
        contexts = [c["content"][:ctx_chars] for c in chunks[:ctx_chunks]] or ["(no context retrieved)"]
        records.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": result["answer"],
            "reference": p.get("ground_truth", ""),
        })
        print(f"  [{'REFUSED' if result['refused'] else 'answered'}] {q[:60]}")
    return records, refusals


def _ensure_vertexai_shim():
    """
    Work around ragas bug #2741: ragas/llms/base.py does
    `from langchain_community.chat_models.vertexai import ChatVertexAI`, a path
    newer langchain-community removed, so `import ragas` crashes before our code
    runs. We register a shim for that submodule. CodeRAG uses Gemini, never
    VertexAI, so the class is only a placeholder and is never instantiated.
    """
    import sys, types
    mod_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(mod_name)
        return                      # already importable, nothing to do
    except Exception:
        pass
    shim = types.ModuleType(mod_name)
    try:
        from langchain_google_vertexai import ChatVertexAI as _CVA
    except Exception:
        class _CVA:                 # placeholder; unused with the Gemini judge
            def __init__(self, *a, **k):
                raise RuntimeError("VertexAI not configured; CodeRAG uses Gemini.")
    shim.ChatVertexAI = _CVA
    sys.modules[mod_name] = shim


def _build_judge():
    """Build the RAGAS judge LLM from config. Pluggable: groq or gemini."""
    from ragas.llms import LangchainLLMWrapper
    cfg = load_settings()["evaluation"]
    provider = cfg.get("judge_provider", "groq")
    model = cfg["judge_model"]

    max_tokens = cfg.get("judge_max_tokens", 8192)
    effort = cfg.get("judge_reasoning_effort")  # groq gpt-oss only

    if provider == "groq":
        from langchain_groq import ChatGroq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY not set (the eval judge). Add it to .env.")
        kwargs = dict(model=model, api_key=key, temperature=0.0, max_tokens=max_tokens)
        if effort:
            kwargs["reasoning_effort"] = effort   # explicit param, not model_kwargs
        llm = ChatGroq(**kwargs)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model,
                                     google_api_key=os.getenv("GOOGLE_API_KEY"),
                                     temperature=0.0,
                                     max_output_tokens=max_tokens)
    else:
        raise ValueError(f"unknown judge_provider: {provider!r} (use 'groq' or 'gemini')")
    print(f"[eval] Judge: {provider} / {model}")
    return LangchainLLMWrapper(llm)


def run_ragas(records):
    """Score faithfulness with RAGAS using the configured judge LLM."""
    _ensure_vertexai_shim()
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness

    judge = _build_judge()
    from ragas.run_config import RunConfig
    dataset = EvaluationDataset.from_list(records)
    # max_workers=1 => one judge call at a time (avoids 429 rate-limit bursts)
    run_config = RunConfig(max_workers=1, timeout=120, max_retries=5, max_wait=90)
    result = evaluate(dataset=dataset, metrics=[Faithfulness()], llm=judge,
                      run_config=run_config)
    return result


def mean_faithfulness(ragas_result):
    """Return (mean, n_scored, n_total). Drops None/NaN but reports the count."""
    try:
        df = ragas_result.to_pandas()
        raw = df["faithfulness"].tolist()
        vals = [v for v in raw if v is not None and v == v]  # drop None/NaN
        mean = sum(vals) / len(vals) if vals else 0.0
        return mean, len(vals), len(raw)
    except Exception:
        return float(ragas_result["faithfulness"]), 1, 1


def gate(score: float, threshold: float) -> int:
    """Return process exit code: 0 = pass, 1 = regression."""
    return 0 if score >= threshold else 1


def main():
    settings = load_settings()["evaluation"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="evaluate only the first N golden pairs (PR smoke run)")
    ap.add_argument("--min-faithfulness", type=float,
                    default=settings["min_faithfulness"],
                    help="regression threshold; build fails below this")
    ap.add_argument("--sleep", type=float, default=4.0,
                    help="seconds to wait between questions (free-tier throttle)")
    args = ap.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set (needed for answer generation).")
        sys.exit(2)
    judge_provider = settings.get("judge_provider", "groq")
    if judge_provider == "groq" and not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set (needed for the eval judge). Get one at console.groq.com.")
        sys.exit(2)

    pairs = load_golden(args.subset)
    print(f"[eval] {len(pairs)} golden pairs | gate: faithfulness >= {args.min_faithfulness}")

    store = build_eval_store()
    try:
        records, refusals = collect_samples(
            store, pairs, sleep_s=args.sleep,
            ctx_chunks=settings.get("judge_context_chunks", 3),
            ctx_chars=settings.get("judge_context_char_limit", 900),
        )
        print("\n[eval] Scoring faithfulness with RAGAS...")
        result = run_ragas(records)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            print("\nERROR: LLM provider rate limit / quota exhausted (HTTP 429).")
            print("  - Wait for the quota to reset, or enable billing, or")
            print("  - run a smaller --subset, raise --sleep, or switch provider in settings.yaml.")
            sys.exit(2)
        raise
    score, n_scored, n_total = mean_faithfulness(result)

    print("\n" + "=" * 60)
    print(f"  Mean faithfulness : {score:.3f}")
    print(f"  Threshold         : {args.min_faithfulness:.3f}")
    print(f"  Scored            : {n_scored}/{n_total}")
    print(f"  Refusals          : {refusals}/{len(records)}")
    print("=" * 60)
    if n_scored < n_total:
        print(f"  WARNING: {n_total - n_scored} sample(s) did not score "
              f"(judge truncation/parse) and were excluded from the mean.")

    code = gate(score, args.min_faithfulness)
    print("PASS ✅" if code == 0 else "FAIL ❌ (faithfulness below threshold)")
    sys.exit(code)


if __name__ == "__main__":
    main()
