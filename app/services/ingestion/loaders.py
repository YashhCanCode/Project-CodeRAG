"""
ingestion/loaders.py

Loads documents for both halves of CodeRAG's domain — codebases AND technical docs.

Sources:
  - Local directory  -> code, markdown/rst/txt, HTML, PDFs, Jupyter notebooks, config
  - GitHub repo      -> cloned, then loaded as a local directory
  - Web page (URL)   -> fetched and converted to text

Each Document carries metadata: {source, language, repo, file_type, (page)}
so citations point back to the exact file (and PDF page).
"""

import os
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# ── File extensions ──────────────────────────────────────────────────────────
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".rb", ".php", ".cs", ".swift", ".kt",
}
# Plain-text technical docs (read directly)
TEXT_DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}
# Config / structured text (useful context in a codebase)
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".env"}
# Need special parsing
PDF_EXTENSIONS = {".pdf"}
HTML_EXTENSIONS = {".html", ".htm"}
NOTEBOOK_EXTENSIONS = {".ipynb"}

ALL_EXTENSIONS = (
    CODE_EXTENSIONS | TEXT_DOC_EXTENSIONS | CONFIG_EXTENSIONS
    | PDF_EXTENSIONS | HTML_EXTENSIONS | NOTEBOOK_EXTENSIONS
)

LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".go": "go",
    ".rs": "rust", ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
    ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".swift": "swift", ".kt": "kotlin",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "text", ".env": "text",
    ".pdf": "pdf", ".html": "html", ".htm": "html", ".ipynb": "python",
}

SKIP_DIRS = {"node_modules", "__pycache__", ".git", "dist", "build",
             "venv", ".venv", ".next", "chroma_db", "tmp_repo"}


# ── Per-format extractors ─────────────────────────────────────────────────────
def _ocr_pdf(path: Path) -> List[tuple]:
    """
    OCR a scanned PDF: render each page to an image (PyMuPDF) and read text
    with Tesseract. Requires: pip install pymupdf pytesseract pillow, and the
    tesseract binary (macOS: `brew install tesseract`).
    """
    try:
        import io
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except Exception as e:
        print(f"  [loader] OCR libraries unavailable ({type(e).__name__}). "
              f"Install with: pip install pymupdf pytesseract pillow "
              f"(plus the tesseract binary — macOS: brew install tesseract).")
        return []

    out = []
    doc = fitz.open(str(path))
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"  [loader] OCR failed on page {i} ({type(e).__name__}); "
                  f"is the tesseract binary installed?")
            text = ""
        if text.strip():
            out.append((text, i))
    print(f"  [loader] OCR extracted text from {len(out)} page(s) of {path.name}")
    return out


def _extract_pdf(path: Path) -> List[tuple]:
    """
    Return [(text, page_number)] per page. Falls back to OCR when the PDF has
    no text layer (i.e. a scanned/image-only PDF).
    """
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append((text, i))

    if not out:
        print(f"  [loader] No text layer in {path.name}; attempting OCR…")
        out = _ocr_pdf(path)
    return out


def _extract_html(raw: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_notebook(raw: str) -> str:
    """Flatten a .ipynb into markdown + code cell text."""
    nb = json.loads(raw)
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        src = "".join(src) if isinstance(src, list) else str(src)
        if not src.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(src)
        else:
            parts.append(src)
    return "\n\n".join(parts)


def _make_doc(content, source, suffix, repo_name, page=None) -> Document:
    meta = {
        "source": source,
        "language": LANGUAGE_MAP.get(suffix, "text"),
        "repo": repo_name,
        "file_type": suffix,
    }
    if page is not None:
        meta["page"] = page
    return Document(page_content=content, metadata=meta)


def _load_file(path: Path, root: Path, repo_name: str) -> List[Document]:
    """Load a single file into one or more Documents (PDFs -> one per page)."""
    suffix = path.suffix.lower()
    rel = str(path.relative_to(root))
    docs: List[Document] = []
    try:
        if suffix in PDF_EXTENSIONS:
            for text, page in _extract_pdf(path):
                docs.append(_make_doc(text, rel, suffix, repo_name, page=page))
        elif suffix in HTML_EXTENSIONS:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = _extract_html(raw)
            if text.strip():
                docs.append(_make_doc(text, rel, suffix, repo_name))
        elif suffix in NOTEBOOK_EXTENSIONS:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = _extract_notebook(raw)
            if text.strip():
                docs.append(_make_doc(text, rel, suffix, repo_name))
        else:  # code / text / config
            content = path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                docs.append(_make_doc(content, rel, suffix, repo_name))
    except Exception as e:
        print(f"  [warn] Could not read {path}: {e}")
    return docs


# ── Public loaders ─────────────────────────────────────────────────────────────
def load_local_directory(directory: str, repo_name: str = "local") -> List[Document]:
    """Load all supported files from a directory, or a single file if a file path is given."""
    docs: List[Document] = []
    root = Path(directory)

    # Allow pointing at a single file directly (e.g. one PDF).
    if root.is_file():
        if root.suffix.lower() in ALL_EXTENSIONS:
            docs = _load_file(root, root.parent, repo_name)
            print(f"[loader] Loaded {len(docs)} documents from file '{directory}'")
        else:
            print(f"[loader] Unsupported file type: {root.suffix}")
        return docs

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Only skip folders *inside* the ingest root (not the clone path prefix
        # like ./tmp_repo, which is part of `root` itself).
        rel_parts = path.relative_to(root).parts
        if any(skip in rel_parts for skip in SKIP_DIRS):
            continue
        if path.suffix.lower() not in ALL_EXTENSIONS:
            continue
        docs.extend(_load_file(path, root, repo_name))

    print(f"[loader] Loaded {len(docs)} documents from '{directory}'")
    return docs


def load_github_repo(repo_url: str, clone_dir: str = "./tmp_repo") -> List[Document]:
    """Clone a GitHub repo and load it as Documents."""
    import git
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_path = os.path.join(clone_dir, repo_name)
    if os.path.exists(clone_path):
        print(f"[loader] Repo already cloned at {clone_path}, using cached copy")
    else:
        print(f"[loader] Cloning {repo_url} -> {clone_path}")
        git.Repo.clone_from(repo_url, clone_path, depth=1)
    return load_local_directory(clone_path, repo_name=repo_name)


def load_web_page(url: str, repo_name: str = "web") -> List[Document]:
    """Fetch a web page and load its text as a single Document."""
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (CodeRAG ingestion)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    text = _extract_html(resp.text)
    if not text.strip():
        print(f"[loader] No extractable text at {url}")
        return []
    doc = Document(
        page_content=text,
        metadata={"source": url, "language": "web", "repo": repo_name, "file_type": ".html"},
    )
    print(f"[loader] Loaded web page: {url}")
    return [doc]
