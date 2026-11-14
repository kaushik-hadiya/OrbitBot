import { useEffect, useState } from "react";
import { getDoc, type KbDoc } from "./api";
import Markdown from "./Markdown";

export default function DocDrawer({ docId, onClose }: { docId: string; onClose: () => void }) {
  const [doc, setDoc] = useState<KbDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDoc(null);
    setError(null);
    getDoc(docId).then(setDoc, (e: Error) => setError(e.message));
  }, [docId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="overlay" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-label={`Source document ${docId}`} onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <div>
            <div className="eyebrow">Source document · {docId}</div>
            <h2>{doc?.title ?? (error ? "Not available" : "Loading…")}</h2>
            {doc && (
              <div className="doc-meta">
                {doc.department && <span>{doc.department}</span>}
                {doc.version != null && <span>v{doc.version}</span>}
                {doc.status && <span className={`status status-${doc.status}`}>{doc.status}</span>}
                {doc.effective_date && <span>effective {doc.effective_date}</span>}
              </div>
            )}
            {doc?.applies_to && <div className="muted small">Applies to: {doc.applies_to}</div>}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        <div className="drawer-body">
          {error && <p className="error">{error}</p>}
          {doc && <Markdown text={doc.content} />}
        </div>
      </aside>
    </div>
  );
}
