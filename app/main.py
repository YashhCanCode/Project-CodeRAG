"""
app/main.py

FastAPI entrypoint for CodeRAG. Wires CORS and the route modules.

Run:  uvicorn app.main:app --reload
Endpoints: GET / · GET /health · POST /ingest · POST /ingest/upload · POST /query
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.routes import health, ingest, query, research, metrics

app = FastAPI(title="CodeRAG", version=__version__)

# CORS so the Next.js frontend (localhost:3000) can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo: any origin; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(research.router)
app.include_router(metrics.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
