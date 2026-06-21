"""
try_retrieval.py  —  CodeRAG test harness

Ingests a source, retrieves top-k chunks (hybrid + rerank), and optionally
generates a cited answer with the LLM.

Usage:
    # retrieval only (no LLM, no API key)
    python try_retrieval.py --repo "https://github.com/owner/name" --query "how is auth handled?"

    # retrieval + LLM answer (needs GOOGLE_API_KEY in .env)
    python try_retrieval.py --repo "https://github.com/owner/name" --query "how is auth handled?" --answer

    # interactive (add --answer to also generate answers)
    python try_retrieval.py --path ./some_repo

    # reuse an already-ingested store
    python try_retrieval.py --query "where is the retry logic?" --no-ingest --answer

Sources: --repo (GitHub), --path (local file/dir), --url (web page).
Retrieval needs only the BGE embedder (+ reranker on first run). --answer
additionally needs the LLM key.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ingestion.loaders import load_local_directory, load_web_page, load_github_repo
from app.services.chunking.chunker import chunk_documents
from app.services.vector_store.store import build_vector_store, load_vector_store
from app.services.retrieval.pipeline import retrieve


def _fmt(v, spec="+.3f"):
    """Format a score that may be None (e.g. a BM25-only chunk has no similarity)."""
    return format(v, spec) if isinstance(v, (int, float)) else "  -  "


def _print_results(query, results):
    print(f"\nQuery: {query!r}")
    if not results:
        print("  (no results — is the store empty?)")
        return
    for i, r in enumerate(results, 1):
        # Per-stage scores so you can see what each retrieval stage contributed.
        print(f"\n  [{i}] score={_fmt(r.get('score'))}  "
              f"rerank_prob={_fmt(r.get('rerank_prob'), '.3f')}  "
              f"sim={_fmt(r.get('similarity'), '.3f')}  "
              f"bm25={_fmt(r.get('bm25_score'), '.2f')}  "
              f"<{r['citation']}>  ({r['language']})")
        snippet = r["content"].strip().replace("\n", "\n      ")
        print("      " + snippet[:280] + ("..." if len(snippet) > 280 else ""))


def _print_answer(query, results):
    """Run the LLM over retrieved chunks and print the cited answer."""
    from app.services.generation.chain import generate_answer  # lazy: only needs the LLM pkg here
    result = generate_answer(query, results)
    print("\n" + "=" * 70)
    if result["refused"]:
        print("REFUSED (citation guard): " + result["answer"])
    else:
        print("ANSWER:\n" + result["answer"])
        print("\nCITATIONS:")
        for c in result["citations"]:
            print(f"  - {c}")
        print(f"\n(chunks used: {result['chunks_used']})")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="local file or directory to ingest")
    ap.add_argument("--url", help="web page URL to ingest")
    ap.add_argument("--repo", help="GitHub repo URL to clone and ingest")
    ap.add_argument("--query", help="single query to run (omit for interactive mode)")
    ap.add_argument("--repo-name", default="local")
    ap.add_argument("--no-ingest", action="store_true",
                    help="skip ingest and use the existing persisted store")
    ap.add_argument("--answer", action="store_true",
                    help="also generate a cited LLM answer (needs GOOGLE_API_KEY)")
    args = ap.parse_args()

    if args.answer:
        # load .env so GOOGLE_API_KEY is available to the generation chain
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

    if args.no_ingest:
        store = load_vector_store()
    else:
        if args.repo:
            docs = load_github_repo(args.repo)
        elif args.url:
            docs = load_web_page(args.url, repo_name=args.repo_name)
        elif args.path:
            docs = load_local_directory(args.path, repo_name=args.repo_name)
        else:
            ap.error("provide --path, --repo, or --url (or --no-ingest)")
        if not docs:
            print("No documents loaded.")
            return
        chunks = chunk_documents(docs)
        store = build_vector_store(chunks)

    def handle(q):
        results = retrieve(store, q)
        _print_results(q, results)
        if args.answer:
            _print_answer(q, results)

    if args.query:
        handle(args.query)
    else:
        mode = "retrieval + answer" if args.answer else "retrieval"
        print(f"\nInteractive {mode}. Type a question (blank line to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            handle(q)


if __name__ == "__main__":
    main()
