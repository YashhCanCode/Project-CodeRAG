"""Ingestion routes: load a source (local / github / web) or upload files."""

import os
import shutil
import tempfile
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.services.ingestion.loaders import (
    load_local_directory, load_github_repo, load_web_page,
)
from app.services.chunking.chunker import chunk_documents
from app.services.vector_store.store import build_vector_store
from app import state

router = APIRouter()


class IngestRequest(BaseModel):
    source_type: str                       # "local" | "github" | "web"
    path: str                              # local path, repo URL, or page URL
    repo_name: Optional[str] = "local"


@router.post("/ingest")
def ingest(req: IngestRequest):
    if req.source_type == "local":
        docs = load_local_directory(req.path, repo_name=req.repo_name)
    elif req.source_type == "github":
        docs = load_github_repo(req.path)
    elif req.source_type == "web":
        docs = load_web_page(req.path, repo_name=req.repo_name)
    else:
        raise HTTPException(400, "source_type must be 'local', 'github', or 'web'")

    if not docs:
        raise HTTPException(400, f"No supported files found at: {req.path}")

    chunks = chunk_documents(docs)
    state.set_store(build_vector_store(chunks))
    return {"status": "ok", "documents": len(docs), "chunks": len(chunks)}


@router.post("/ingest/upload")
async def ingest_upload(files: List[UploadFile] = File(...)):
    """Ingest uploaded files (PDF, code, markdown, …). Replaces the current store."""
    from app.services.ingestion.loaders import ALL_EXTENSIONS

    tmpdir = tempfile.mkdtemp(prefix="coderag_upload_")
    saved_names: List[str] = []
    try:
        for f in files:
            name = os.path.basename(f.filename or "file")
            data = await f.read()                 # async read (robust for uploads)
            with open(os.path.join(tmpdir, name), "wb") as out:
                out.write(data)
            saved_names.append(name)

        docs = load_local_directory(tmpdir, repo_name="upload")
        if not docs:
            supported = ", ".join(sorted(ALL_EXTENSIONS))
            raise HTTPException(
                400,
                f"No extractable text found in: {saved_names}. "
                f"Supported file types: {supported}. "
                f"Note: image-only/scanned PDFs have no text layer, and formats like "
                f".docx/.xlsx/.csv/images are not supported.",
            )
        chunks = chunk_documents(docs)
        state.set_store(build_vector_store(chunks))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {"status": "ok", "files": len(saved_names),
            "documents": len(docs), "chunks": len(chunks)}
