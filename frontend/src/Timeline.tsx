import type { ReactNode } from "react";
import type { Step } from "./api";

// One line per graph node, in the order they ran. "warn" marks the moments Self-RAG corrects itself
// (a revision, a query rewrite, an unverified claim): the part a plain RAG pipeline doesn't have.
type Tone = "ok" | "warn" | "info" | "fail";
type Line = { tone: Tone; label: string; detail?: ReactNode; tag?: string };

const ROUTES: Record<string, string> = {
  knowledge_base: "search the knowledge base",
  action: "use tools (agent)",
  small_talk: "small talk, no search",
  out_of_scope: "out of scope",
  distress: "wellbeing support",
};

const pct = (p: unknown) => (typeof p === "number" ? `${Math.round(p * 100)}%` : undefined);

function describe(step: Step, question: string): Line | null {
  const d = step.data ?? {};
  switch (step.node) {
    case "start_turn":
      return d.standalone_question && d.standalone_question !== question
        ? { tone: "info", label: "Rewrote the follow-up to stand alone", detail: <q>{d.standalone_question}</q> }
        : null;
    case "decide_retrieval":
      if (d.restricted)
        return { tone: "fail", label: "Safety check: refused", detail: `flag: ${d.restricted}`, tag: "Retrieve?" };
      return {
        tone: "ok",
        label: `Decided to ${ROUTES[d.route] ?? d.route}`,
        detail: d.probability != null ? `decision model confidence ${pct(d.probability)}` : d.reason,
        tag: "Retrieve?",
      };
    case "plan_queries": {
      const parts: string[] = d.sub_questions ?? [];
      return parts.length > 1
        ? {
            tone: "info",
            label: `Split into ${parts.length} searches`,
            detail: (
              <ol>
                {parts.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ol>
            ),
          }
        : { tone: "info", label: "Planned one search" };
    }
    case "retrieve": {
      const chunks: { doc_id: string; status: string }[] = d.chunks ?? [];
      const docs = [...new Set(chunks.map((c) => c.doc_id))];
      return {
        tone: "info",
        label: `Retrieved ${chunks.length} chunks`,
        detail: `hybrid BM25 + vector search, reranked · ${docs.join(", ")}`,
      };
    }
    case "grade_documents": {
      const grades: { doc_id: string; title: string; relevant: boolean; reason: string }[] = d.grades ?? [];
      const kept = grades.filter((g) => g.relevant).length;
      return {
        tone: kept ? "ok" : "warn",
        label: `Graded relevance: kept ${kept} of ${grades.length}`,
        tag: "IsRel",
        detail: (
          <ul className="grades">
            {grades.map((g, i) => (
              <li key={i} className={g.relevant ? "kept" : "dropped"}>
                <span aria-hidden>{g.relevant ? "✓" : "✗"}</span> <b>{g.doc_id}</b> {g.title}{" "}
                <span className="muted">({g.reason})</span>
              </li>
            ))}
          </ul>
        ),
      };
    }
    case "generate":
      return { tone: "info", label: "Wrote an answer from the kept chunks, with citations" };
    case "check_support": {
      const claims: string[] = d.unsupported_claims ?? [];
      return d.support === "fully"
        ? { tone: "ok", label: "Every claim is supported by the sources", tag: "IsSup" }
        : {
            tone: "warn",
            label: `Support: ${d.support}, ${claims.length} unsupported claim${claims.length === 1 ? "" : "s"}`,
            tag: "IsSup",
            detail: (
              <ul>
                {claims.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            ),
          };
    }
    case "revise_answer":
      return { tone: "warn", label: "Revised the answer to remove the unsupported claims" };
    case "flag_unsupported":
      return { tone: "warn", label: "Still unverified after a revision: flagged in the answer" };
    case "check_useful":
      return d.useful
        ? { tone: "ok", label: "The answer addresses the question", tag: "IsUse", detail: d.reason }
        : { tone: "warn", label: "The answer doesn't help yet", tag: "IsUse", detail: d.reason };
    case "rewrite_query":
      return { tone: "warn", label: "Rewrote the search query and searched again", detail: <q>{d.new_query}</q> };
    case "suggest_contact":
      return { tone: "info", label: "Pointed to the owner of the closest document" };
    case "generate_direct":
      return { tone: "info", label: "Replied directly, without company facts" };
    case "decline_out_of_scope":
      return { tone: "info", label: "Fixed reply: outside what OrbitBot covers" };
    case "decline_restricted":
      return { tone: "fail", label: "Fixed refusal, no LLM call" };
    case "support_wellbeing":
      return { tone: "info", label: "Fixed wellbeing reply with support contacts" };
    case "agent":
      if (d.final) return { tone: "info", label: "Agent wrote the final answer" };
      return {
        tone: "info",
        label: "Agent called tools",
        detail: (
          <ul>
            {(d.tool_calls ?? []).map((c: { name: string; args: object }, i: number) => (
              <li key={i}>
                <code>
                  {c.name}({Object.keys(c.args).length ? JSON.stringify(c.args) : ""})
                </code>
              </li>
            ))}
          </ul>
        ),
      };
    case "tools":
      return {
        tone: "info",
        label: `Tool result${(d.results ?? []).length > 1 ? "s" : ""}`,
        detail: (
          <ul>
            {(d.results ?? []).map((r: { name: string; preview: string }, i: number) => (
              <li key={i}>
                <code>{r.name}</code> <span className="muted">{r.preview.slice(0, 140)}…</span>
              </li>
            ))}
          </ul>
        ),
      };
    default:
      return { tone: "info", label: step.node };
  }
}

export default function Timeline({ steps, question, running }: { steps: Step[]; question: string; running: boolean }) {
  const lines = steps.map((s) => ({ s, line: describe(s, question) })).filter((x) => x.line);
  return (
    <ol className="timeline">
      {lines.map(({ s, line }, i) => (
        <li key={i} className={`tl tl-${line!.tone}`}>
          <span className="dot" aria-hidden />
          <div className="tl-body">
            <div className="tl-head">
              <span>{line!.label}</span>
              {line!.tag && <span className="tag">{line!.tag}</span>}
              <span className="time">{s.t.toFixed(1)}s</span>
            </div>
            {line!.detail && <div className="tl-detail">{line!.detail}</div>}
          </div>
        </li>
      ))}
      {running && (
        <li className="tl tl-pending">
          <span className="dot" aria-hidden />
          <div className="tl-body">
            <div className="tl-head">
              <span className="shimmer">Working…</span>
            </div>
          </div>
        </li>
      )}
    </ol>
  );
}
