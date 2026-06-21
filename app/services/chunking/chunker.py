"""
ingestion/chunker.py

Code-aware chunking: splits on function/class boundaries first,
falls back to paragraph then character splitting.

Why this matters: generic splitters cut mid-function, breaking
the semantic unit. A chunk containing half a class is useless for
code Q&A. We split on `\nclass ` and `\ndef ` first.

Sizing is measured in TOKENS (spec: 500-800 tokens, ~100-token overlap)
via tiktoken. If tiktoken's encoding can't be downloaded (offline / CI),
we fall back to a character-based splitter at ~4 chars/token and warn.

Each chunk inherits metadata from its parent document and adds:
  - chunk_index: position in the document
  - start_line / end_line: line range (for citations), located with a
    forward-moving cursor so repeated/duplicate code maps to the right copy.
"""

from typing import List, Tuple
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.paths import load_settings

# Languages that get class/function-boundary separators.
CODE_LANGUAGES = {
    "python", "javascript", "typescript", "go",
    "rust", "java", "cpp", "c", "ruby", "php", "csharp",
}


@lru_cache(maxsize=1)
def _tiktoken_available() -> bool:
    """True if a tiktoken encoding can actually be loaded (needs one-time download)."""
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception as e:
        print(f"  [chunker] tiktoken unavailable ({type(e).__name__}); "
              f"falling back to character-based sizing (~4 chars/token).")
        return False


def _separators_for(language: str) -> List[str]:
    if language in CODE_LANGUAGES:
        # Split at class/function boundaries first — keeps logical units together.
        return ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
    # Markdown / RST / plain text — heading- and paragraph-first.
    return ["\n## ", "\n### ", "\n\n", "\n", " ", ""]


def _get_splitter_for_language(language: str, chunk_size: int, chunk_overlap: int):
    """
    RecursiveCharacterTextSplitter tuned for the language.
    chunk_size/chunk_overlap are TOKENS when tiktoken is available,
    otherwise CHARACTERS scaled by ~4 chars/token.
    """
    separators = _separators_for(language)

    if _tiktoken_available():
        # Token-counted: chunk_size/overlap are real tokens.
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            is_separator_regex=False,
        )

    # Offline fallback: approximate tokens with characters.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=chunk_overlap * 4,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def _locate(full_text: str, chunk_text: str, cursor: int) -> Tuple[int, int]:
    """
    Find chunk_text at or after `cursor`. Returns (start_char, end_char).
    Searching forward from the previous chunk's start makes duplicate code
    resolve to the correct occurrence (the old global find() always returned
    the first copy).
    """
    start = full_text.find(chunk_text, cursor)
    if start == -1:
        # Splitter may have normalized whitespace; retry from the top as a fallback.
        start = full_text.find(chunk_text)
    if start == -1:
        return (-1, -1)
    return (start, start + len(chunk_text))


def _line_range(full_text: str, start_char: int, end_char: int) -> Tuple[int, int]:
    if start_char < 0:
        return (0, 0)
    start_line = full_text.count("\n", 0, start_char) + 1
    end_line = full_text.count("\n", 0, end_char) + 1
    return (start_line, end_line)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into chunks, each carrying full metadata
    including an accurate line range and a stable chunk id for citations.
    """
    settings = load_settings()
    chunk_size = settings["chunking"]["chunk_size"]
    chunk_overlap = settings["chunking"]["chunk_overlap"]

    all_chunks = []

    for doc in documents:
        language = doc.metadata.get("language", "text")
        splitter = _get_splitter_for_language(language, chunk_size, chunk_overlap)
        source = doc.metadata.get("source", "unknown")

        raw_chunks = splitter.split_text(doc.page_content)

        cursor = 0
        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            start_char, end_char = _locate(doc.page_content, chunk_text, cursor)
            if start_char >= 0:
                # Advance cursor so the next (overlapping) chunk is found forward.
                cursor = start_char + 1
            start_line, end_line = _line_range(doc.page_content, start_char, end_char)

            page = doc.metadata.get("page")
            if page is not None:
                # PDFs: line numbers are within-page, so cite the page too.
                citation = f"{source} p.{page} (lines {start_line}-{end_line})"
                chunk_id = f"{source}:p{page}:{i}:{start_line}-{end_line}"
            else:
                citation = f"{source} lines {start_line}-{end_line}"
                chunk_id = f"{source}:{i}:{start_line}-{end_line}"

            chunk = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_id": chunk_id,      # stable => idempotent re-ingest
                    "citation": citation,      # ready to use in prompts
                }
            )
            all_chunks.append(chunk)

    print(f"[chunker] {len(documents)} documents -> {len(all_chunks)} chunks")
    return all_chunks
