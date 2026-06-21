"""Offline unit test for the chunker's citation metadata."""

from langchain_core.documents import Document
from app.services.chunking.chunker import chunk_documents


def test_chunks_carry_citation_and_id():
    doc = Document(
        page_content="def alpha():\n    return 1\n\ndef beta():\n    return 2\n",
        metadata={"source": "svc.py", "language": "python"},
    )
    chunks = chunk_documents([doc])
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.metadata.get("citation")
        assert c.metadata.get("chunk_id")
        assert "svc.py" in c.metadata["citation"]
