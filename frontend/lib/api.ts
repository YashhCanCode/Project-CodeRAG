// Minimal typed client for the CodeRAG FastAPI backend.

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type IngestResult = {
  status: string;
  documents: number;
  chunks: number;
  files?: number;
};

export type Chunk = {
  citation: string;
  source: string;
  language: string;
  score: number | null;
  content: string;
  preview: string;
};

export type QueryResult = {
  question: string;
  retrieved: Chunk[];
  answer: string | null;
  citations: string[];
  refused: boolean;
  chunks_used: number;
  answer_error?: string;
};

async function asError(res: Response): Promise<never> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body?.detail ?? detail;
  } catch {
    /* ignore */
  }
  throw new Error(`${res.status}: ${detail}`);
}

export async function ingestUrl(
  sourceType: "github" | "web",
  path: string
): Promise<IngestResult> {
  const res = await fetch(`${API}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_type: sourceType, path }),
  });
  if (!res.ok) return asError(res);
  return res.json();
}

export async function ingestFiles(files: FileList): Promise<IngestResult> {
  const form = new FormData();
  Array.from(files).forEach((f) => form.append("files", f));
  const res = await fetch(`${API}/ingest/upload`, { method: "POST", body: form });
  if (!res.ok) return asError(res);
  return res.json();
}

export async function search(question: string): Promise<QueryResult> {
  const res = await fetch(`${API}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, generate: true }),
  });
  if (!res.ok) return asError(res);
  return res.json();
}

// ── Research landscape (arXiv) ────────────────────────────────────────────
export type Paper = {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  url: string;
  pdf_url: string;
  published: string;
  categories: string[];
  problem: string;
  method: string;
  results: string;
  contribution: string;
  cluster: string;
};

export type Synthesis = {
  overview: string;
  clusters: { theme: string; paper_ids: string[] }[];
  open_problems: string[];
  tensions: string[];
};

export type Landscape = {
  topic: string;
  papers: Paper[];
  synthesis: Synthesis;
};

export async function research(topic: string, maxPapers = 8): Promise<Landscape> {
  const res = await fetch(`${API}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, max_papers: maxPapers }),
  });
  if (!res.ok) return asError(res);
  return res.json();
}
