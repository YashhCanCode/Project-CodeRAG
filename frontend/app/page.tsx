"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  ingestUrl, ingestFiles, search, research,
  type QueryResult, type Chunk, type Paper, type Landscape,
} from "@/lib/api";

type Tab = "upload" | "github" | "web";
type Mode = "search" | "research";

const PRISM_LANG: Record<string, string> = {
  python: "python", javascript: "javascript", typescript: "typescript",
  go: "go", rust: "rust", java: "java", cpp: "cpp", c: "c", ruby: "ruby",
  php: "php", csharp: "csharp", swift: "swift", kotlin: "kotlin",
  markdown: "markdown", json: "json", yaml: "yaml", toml: "toml",
  html: "markup", web: "markup", rst: "rest", text: "text", pdf: "text",
};

const basename = (p: string) => p.split("/").pop() || p;

// ── Ingest panel ────────────────────────────────────────────────────────────
function IngestPanel({ onIngested }: { onIngested: () => void }) {
  const [tab, setTab] = useState<Tab>("upload");
  const [urlValue, setUrlValue] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const tabs: { id: Tab; label: string }[] = [
    { id: "upload", label: "Upload" }, { id: "github", label: "GitHub repo" }, { id: "web", label: "Web page" },
  ];

  async function handleIngest() {
    setError(null); setNote(null); setIngesting(true);
    try {
      let r;
      if (tab === "upload") {
        const files = fileRef.current?.files;
        if (!files || files.length === 0) throw new Error("Choose at least one file.");
        r = await ingestFiles(files);
      } else {
        if (!urlValue.trim()) throw new Error("Enter a URL.");
        r = await ingestUrl(tab, urlValue.trim());
      }
      setNote(`Indexed ${r.documents} documents → ${r.chunks} chunks.`);
      onIngested();
    } catch (e: any) { setError(e.message ?? "Ingestion failed."); }
    finally { setIngesting(false); }
  }

  return (
    <section className="rounded-2xl border border-neutral-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex gap-1 rounded-lg bg-neutral-100 p-1 text-sm">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => { setTab(t.id); setError(null); }}
            className={`flex-1 rounded-md px-3 py-1.5 font-medium transition ${
              tab === t.id ? "bg-white text-neutral-900 shadow-sm" : "text-neutral-500 hover:text-neutral-700"}`}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex flex-col gap-2 sm:flex-row">
        {tab === "upload" ? (
          <input key="ingest-file" ref={fileRef} type="file" multiple
            accept=".pdf,.md,.mdx,.rst,.txt,.py,.js,.ts,.tsx,.jsx,.go,.rs,.java,.cpp,.c,.h,.rb,.php,.cs,.html,.ipynb,.json,.yaml,.yml,.toml"
            className="block w-full text-sm text-neutral-600 file:mr-3 file:rounded-lg file:border-0 file:bg-neutral-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-neutral-700" />
        ) : (
          <input key="ingest-url" type="text" value={urlValue}
            onChange={(e) => setUrlValue(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleIngest()}
            placeholder={tab === "github" ? "https://github.com/owner/repo" : "https://docs.example.com/page"}
            className="w-full rounded-lg border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-900" />
        )}
        <button onClick={handleIngest} disabled={ingesting}
          className="shrink-0 rounded-lg bg-neutral-900 px-5 py-2 text-sm font-medium text-white transition hover:bg-neutral-700 disabled:opacity-50">
          {ingesting ? "Indexing…" : "Ingest"}
        </button>
      </div>
      {note && <p className="mt-2 text-sm text-emerald-600">{note}</p>}
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </section>
  );
}

// ── Slide-in panel shell (left) ──────────────────────────────────────────────
function SlideOver({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  const [show, setShow] = useState(false);
  useEffect(() => { const id = requestAnimationFrame(() => setShow(true)); return () => cancelAnimationFrame(id); }, []);
  const close = useCallback(() => { setShow(false); const t = setTimeout(onClose, 300); return () => clearTimeout(t); }, [onClose]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey);
  }, [close]);
  return (
    <div className="fixed inset-0 z-50">
      <div className={`absolute inset-0 bg-neutral-900/30 transition-opacity duration-300 ${show ? "opacity-100" : "opacity-0"}`} onClick={close} />
      <div className={`absolute left-0 top-0 flex h-full w-full max-w-md flex-col bg-white shadow-2xl transition-transform duration-300 ease-out ${show ? "translate-x-0" : "-translate-x-full"}`}>
        {typeof children === "function" ? (children as any)(close) : children}
      </div>
    </div>
  );
}

// ── Chunk detail (search mode) ───────────────────────────────────────────────
function ChunkPanel({ chunk, onClose }: { chunk: Chunk; onClose: () => void }) {
  const lang = PRISM_LANG[chunk.language] || "text";
  return (
    <SlideOver onClose={onClose}>
      <div className="flex items-start justify-between gap-3 border-b border-neutral-200 px-5 py-3">
        <div className="min-w-0">
          <p className="truncate font-medium text-neutral-900">{chunk.source}</p>
          <p className="truncate font-mono text-xs text-neutral-500">{chunk.citation}</p>
        </div>
        <button onClick={onClose} className="shrink-0 rounded-lg px-2 py-1 text-sm text-neutral-400 hover:bg-neutral-100 hover:text-neutral-700">Esc ✕</button>
      </div>
      <div className="flex-1 overflow-auto p-1 text-sm">
        <SyntaxHighlighter language={lang} style={oneLight} customStyle={{ margin: 0, background: "transparent", fontSize: "12.5px" }} wrapLongLines>
          {chunk.content}
        </SyntaxHighlighter>
      </div>
    </SlideOver>
  );
}

// ── Paper detail (research mode) ─────────────────────────────────────────────
function Field({ label, value }: { label: string; value: string }) {
  if (!value) return null;
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">{label}</p>
      <p className="mt-0.5 text-sm leading-relaxed text-neutral-800">{value}</p>
    </div>
  );
}

function PaperPanel({ paper, onClose }: { paper: Paper; onClose: () => void }) {
  return (
    <SlideOver onClose={onClose}>
      <div className="flex items-start justify-between gap-3 border-b border-neutral-200 px-5 py-3">
        <div className="min-w-0">
          <p className="font-medium leading-snug text-neutral-900">{paper.title}</p>
          <p className="mt-0.5 truncate text-xs text-neutral-500">
            {paper.authors.slice(0, 3).join(", ")}{paper.authors.length > 3 ? " et al." : ""} · {paper.published}
          </p>
        </div>
        <button onClick={onClose} className="shrink-0 rounded-lg px-2 py-1 text-sm text-neutral-400 hover:bg-neutral-100 hover:text-neutral-700">Esc ✕</button>
      </div>
      <div className="flex-1 space-y-4 overflow-auto px-5 py-4">
        <Field label="Problem" value={paper.problem} />
        <Field label="Method" value={paper.method} />
        <Field label="Results" value={paper.results} />
        <Field label="Contribution" value={paper.contribution} />
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">Abstract</p>
          <p className="mt-0.5 text-sm leading-relaxed text-neutral-600">{paper.abstract}</p>
        </div>
        <a href={paper.url} target="_blank" rel="noreferrer"
          className="inline-block rounded-lg bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700">
          View on arXiv ↗
        </a>
      </div>
    </SlideOver>
  );
}

function PaperCard({ paper, onClick }: { paper: Paper; onClick: () => void }) {
  return (
    <button onClick={onClick}
      className="flex flex-col gap-2 rounded-xl border border-neutral-200 bg-white p-4 text-left shadow-sm transition hover:border-neutral-400 hover:shadow">
      <p className="font-medium leading-snug text-neutral-900 line-clamp-2">{paper.title}</p>
      <p className="text-xs text-neutral-400">
        {paper.authors.slice(0, 2).join(", ")}{paper.authors.length > 2 ? " et al." : ""} · {paper.published}
      </p>
      {paper.contribution && <p className="line-clamp-3 text-sm text-neutral-600">{paper.contribution}</p>}
    </button>
  );
}

// ── Research landscape view ──────────────────────────────────────────────────
function ResearchView() {
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [land, setLand] = useState<Landscape | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Paper | null>(null);

  const run = useCallback(async () => {
    const t = topic.trim();
    if (!t || loading) return;
    setLoading(true); setError(null);
    try { setLand(await research(t, 8)); }
    catch (e: any) { setError(e.message ?? "Research failed."); setLand(null); }
    finally { setLoading(false); }
  }, [topic, loading]);

  const clusters = land
    ? Array.from(new Set(land.papers.map((p) => p.cluster || "Other")))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <input type="text" value={topic} onChange={(e) => setTopic(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="Enter a research topic (e.g. retrieval-augmented generation)…"
          className="w-full rounded-xl border border-neutral-300 bg-white px-4 py-3 text-base shadow-sm outline-none focus:border-neutral-900" />
        <button onClick={run} disabled={loading || !topic.trim()}
          className="shrink-0 rounded-xl bg-neutral-900 px-6 text-sm font-medium text-white transition hover:bg-neutral-700 disabled:opacity-50">
          {loading ? "Mapping…" : "Map research"}
        </button>
      </div>
      {loading && <p className="text-sm text-neutral-400">Searching arXiv, reranking, and synthesizing… this takes a moment.</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {land && (
        <div className="space-y-6">
          {land.synthesis.overview && (
            <section className="rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">Landscape</p>
              <p className="text-sm leading-relaxed text-neutral-800">{land.synthesis.overview}</p>
            </section>
          )}

          <div className="grid gap-4 sm:grid-cols-2">
            {land.synthesis.open_problems?.length > 0 && (
              <section className="rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">Open problems</p>
                <ul className="list-disc space-y-1 pl-4 text-sm text-neutral-700">
                  {land.synthesis.open_problems.map((x, i) => <li key={i}>{x}</li>)}
                </ul>
              </section>
            )}
            {land.synthesis.tensions?.length > 0 && (
              <section className="rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">Tensions</p>
                <ul className="list-disc space-y-1 pl-4 text-sm text-neutral-700">
                  {land.synthesis.tensions.map((x, i) => <li key={i}>{x}</li>)}
                </ul>
              </section>
            )}
          </div>

          {clusters.map((theme) => (
            <section key={theme} className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">{theme}</p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {land.papers.filter((p) => (p.cluster || "Other") === theme).map((p) => (
                  <PaperCard key={p.id} paper={p} onClick={() => setSelected(p)} />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}

      {selected && <PaperPanel paper={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

// ── Search view ──────────────────────────────────────────────────────────────
function ResultCard({ chunk, onClick }: { chunk: Chunk; onClick: () => void }) {
  const score = typeof chunk.score === "number" ? Math.round(chunk.score * 100) : null;
  return (
    <button onClick={onClick}
      className="group flex flex-col gap-2 rounded-xl border border-neutral-200 bg-white p-4 text-left shadow-sm transition hover:border-neutral-400 hover:shadow">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-medium text-neutral-900">{basename(chunk.source)}</span>
        <span className="shrink-0 rounded-full bg-neutral-100 px-2 py-0.5 text-[11px] font-medium text-neutral-500">{chunk.language || "text"}</span>
      </div>
      <p className="line-clamp-3 font-mono text-xs leading-relaxed text-neutral-500">{chunk.preview.trim()}</p>
      <div className="mt-auto flex items-center justify-between pt-1">
        <span className="truncate font-mono text-[11px] text-neutral-400">{chunk.citation}</span>
        {score !== null && <span className="shrink-0 text-[11px] font-medium text-neutral-400">{score}% match</span>}
      </div>
    </button>
  );
}

function SearchView() {
  const [hasCorpus, setHasCorpus] = useState(false);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Chunk | null>(null);

  const runSearch = useCallback(async () => {
    const question = q.trim();
    if (!question || loading) return;
    setLoading(true); setError(null);
    try { setResult(await search(question)); }
    catch (e: any) { setError(e.message ?? "Search failed."); setResult(null); }
    finally { setLoading(false); }
  }, [q, loading]);

  return (
    <div className="space-y-6">
      <IngestPanel onIngested={() => setHasCorpus(true)} />
      <div className="flex gap-2">
        <input type="text" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runSearch()}
          placeholder={hasCorpus ? "Search or ask anything…" : "Ingest a source above, then search…"}
          className="w-full rounded-xl border border-neutral-300 bg-white px-4 py-3 text-base shadow-sm outline-none focus:border-neutral-900" />
        <button onClick={runSearch} disabled={loading || !q.trim()}
          className="shrink-0 rounded-xl bg-neutral-900 px-6 text-sm font-medium text-white transition hover:bg-neutral-700 disabled:opacity-50">
          {loading ? "Searching…" : "Search"}
        </button>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {result && (
        <div className="space-y-5">
          <section className="rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">Answer</p>
            {result.answer ? (
              <>
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-800">{result.answer}</p>
                {result.citations.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {result.citations.map((c, i) => (
                      <span key={i} className="rounded-full bg-neutral-100 px-2 py-0.5 font-mono text-[11px] text-neutral-500">{c}</span>
                    ))}
                  </div>
                )}
              </>
            ) : result.refused ? (
              <p className="text-sm text-amber-700">Insufficient evidence in the retrieved sources to answer confidently.</p>
            ) : (
              <p className="text-sm text-amber-700">{result.answer_error ?? "Answer unavailable."} The sources below were still retrieved.</p>
            )}
          </section>
          <section className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">Sources ({result.retrieved.length})</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {result.retrieved.map((c, i) => <ResultCard key={i} chunk={c} onClick={() => setSelected(c)} />)}
            </div>
          </section>
        </div>
      )}
      {selected && <ChunkPanel chunk={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────
export default function Home() {
  const [mode, setMode] = useState<Mode>("search");
  const modes: { id: Mode; label: string }[] = [
    { id: "search", label: "Search docs & code" },
    { id: "research", label: "Research landscape" },
  ];
  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col gap-6 px-5 py-10">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">CodeRAG</h1>
        <p className="text-sm text-neutral-500">
          Search your codebases and technical docs — or map a research topic across arXiv.
        </p>
      </header>

      <div className="flex gap-1 rounded-lg bg-neutral-100 p-1 text-sm">
        {modes.map((m) => (
          <button key={m.id} onClick={() => setMode(m.id)}
            className={`flex-1 rounded-md px-3 py-1.5 font-medium transition ${
              mode === m.id ? "bg-white text-neutral-900 shadow-sm" : "text-neutral-500 hover:text-neutral-700"}`}>
            {m.label}
          </button>
        ))}
      </div>

      {mode === "search" ? <SearchView /> : <ResearchView />}

      <footer className="mt-auto pt-4 text-center text-xs text-neutral-400">
        CodeRAG · hybrid retrieval + reranking + citation-enforced answers · arXiv research synthesis
      </footer>
    </main>
  );
}
