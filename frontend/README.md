# CodeRAG — Frontend

A minimal Next.js + Tailwind UI for CodeRAG. Ingest a PDF/files, a GitHub repo,
or a web page, then ask questions and get answers grounded in your sources with
citations.

## Run

You need **two** processes: the FastAPI backend and this frontend.

### 1. Backend (from the project root)
```bash
pip install -r requirements.txt        # includes python-multipart for uploads
uvicorn app.main:app --reload               # serves http://localhost:8000
```
Make sure `.env` has `GOOGLE_API_KEY` (generation needs it).

### 2. Frontend (from this folder)
```bash
npm install
cp .env.local.example .env.local       # points at http://localhost:8000
npm run dev                            # opens http://localhost:3000
```

## Use
1. Pick a tab — Upload, GitHub repo, or Web page — and click **Ingest**.
2. Wait for "Indexed N documents → M chunks".
3. Ask questions in the chat. Answers show their source citations; if there's
   not enough evidence, the citation guard makes it refuse instead of guessing.

## Notes
- First query downloads the reranker (~1.1GB) on the backend; later queries are fast.
- Each ingest replaces the current corpus (one corpus at a time).
- The API base URL is configurable via `NEXT_PUBLIC_API_URL` in `.env.local`.
