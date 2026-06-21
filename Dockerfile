# CodeRAG backend (FastAPI). Heavy ML deps (torch, chromadb) make this image large.
FROM python:3.11-slim

WORKDIR /code

# System deps for some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY prompts/ ./prompts/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
