const GITHUB_URL = import.meta.env.VITE_GITHUB_URL as string | undefined;

const STATS = [
  { value: "0.99", label: "faithfulness", note: "answers grounded in the sources (50-question set)" },
  { value: "68/68", label: "safety cases", note: "refusals, prompt leaks, privacy, fabricated actions" },
  { value: "0.96", label: "answer relevancy", note: "DeepEval, whole pipeline" },
  { value: "6.8 s", label: "median latency", note: "down from 10.8 s by moving checks to a decision model" },
];

const CHECKS = [
  ["Retrieve?", "Does this message need the knowledge base, a tool, or neither?"],
  ["IsRel", "Is each retrieved chunk relevant, current and for the right region?"],
  ["IsSup", "Is every claim in the answer supported by the chunks? If not, revise it."],
  ["IsUse", "Does the answer actually help? If not, rewrite the search and try again."],
];

export default function About({ onClose }: { onClose: () => void }) {
  return (
    <div className="overlay center" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="About OrbitBot" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <div>
            <div className="eyebrow">About this demo</div>
            <h2>A RAG assistant that checks its own work</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        <div className="modal-body">
          <p>
            OrbitBot answers employee questions for <b>Orbitra Technologies</b>, a fictional company with 46 policy
            documents (HR, IT, finance, security…), including deliberate traps: a superseded policy version,
            region-specific rules and questions that need two documents. It follows the{" "}
            <a href="https://arxiv.org/abs/2310.11511" target="_blank" rel="noreferrer">
              Self-RAG
            </a>{" "}
            idea: instead of retrieve-then-answer, it reflects at four points and corrects itself.
          </p>
          <ul className="checks">
            {CHECKS.map(([tag, text]) => (
              <li key={tag}>
                <span className="tag">{tag}</span> {text}
              </li>
            ))}
          </ul>
          <p>
            Actions (“how many leaves do I have?”, “draft a ticket”) go to a tool-calling agent instead. Pick a
            persona at the top to try personal questions; all people and data are synthetic.
          </p>
          <div className="stats">
            {STATS.map((s) => (
              <div key={s.label} className="stat">
                <div className="stat-value">{s.value}</div>
                <div className="stat-label">{s.label}</div>
                <div className="muted small">{s.note}</div>
              </div>
            ))}
          </div>
          <p className="muted small">
            Built with LangGraph, LangChain, hybrid BM25 + vector search (Chroma, bge-small) with a cross-encoder
            reranker, an open-weight LLM via OpenRouter, and a decision model for the yes/no checks. FastAPI streams
            each step to this React UI.
          </p>
          {GITHUB_URL && (
            <a className="btn" href={GITHUB_URL} target="_blank" rel="noreferrer">
              Source code on GitHub
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
