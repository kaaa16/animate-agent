"use client";

import { FormEvent, useState } from "react";

type DocumentBlock = {
  id: string;
  type: "paragraph" | "code" | "list" | "image";
  text: string;
  language?: string | null;
  source_ref?: string | null;
};

type Section = {
  id: string;
  title: string;
  level: number;
  blocks: DocumentBlock[];
};

type DocumentIR = {
  document_id: string;
  title: string;
  sections: Section[];
};

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [url, setUrl] = useState(
    "https://docs.manim.community/en/stable/tutorials/quickstart.html",
  );
  const [document, setDocument] = useState<DocumentIR | null>(null);
  const [activeSectionId, setActiveSectionId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const activeSection = document?.sections.find((section) => section.id === activeSectionId);

  async function parseDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}/api/documents/from-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(payload?.detail ?? `Request failed (${response.status})`);
      }
      const nextDocument = (await response.json()) as DocumentIR;
      setDocument(nextDocument);
      setActiveSectionId(nextDocument.sections[0]?.id ?? null);
    } catch (reason) {
      setDocument(null);
      setActiveSectionId(null);
      setError(reason instanceof Error ? reason.message : "Could not parse this document.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">把世界画出来 · Milestone 1</p>
        <h1>Documentation Parser</h1>
        <p>Turn an official documentation page into a clean, inspectable DocumentIR.</p>
      </header>

      <form onSubmit={parseDocument}>
        <label htmlFor="documentation-url">Documentation URL</label>
        <div className="formRow">
          <input
            id="documentation-url"
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://docs.example.com/quickstart"
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? "Parsing…" : "Parse Document"}
          </button>
        </div>
      </form>

      {error && <p className="error" role="alert">{error}</p>}

      {document && (
        <section className="result" aria-live="polite">
          <div className="resultHeader">
            <p>Document title</p>
            <h2>{document.title}</h2>
            <code>{document.document_id}</code>
          </div>
          <div className="browser">
            <nav aria-label="Document sections">
              <h3>Sections</h3>
              {document.sections.map((section) => (
                <button
                  className={section.id === activeSectionId ? "active" : ""}
                  key={section.id}
                  onClick={() => setActiveSectionId(section.id)}
                  type="button"
                >
                  {section.title}
                </button>
              ))}
            </nav>
            <article>
              {activeSection ? (
                <>
                  <h3>{activeSection.title}</h3>
                  {activeSection.blocks.map((block) => {
                    if (block.type === "code") {
                      return <pre key={block.id}><code>{block.text}</code></pre>;
                    }
                    if (block.type === "list") {
                      return <ul key={block.id}>{block.text.split("\n").map((item) => <li key={item}>{item}</li>)}</ul>;
                    }
                    if (block.type === "image" && block.source_ref) {
                      return <figure key={block.id}><img src={block.source_ref} alt={block.text} /></figure>;
                    }
                    return <p key={block.id}>{block.text}</p>;
                  })}
                </>
              ) : (
                <p>This document has no extractable sections.</p>
              )}
            </article>
          </div>
        </section>
      )}
    </main>
  );
}
