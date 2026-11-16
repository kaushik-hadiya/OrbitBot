import { useCallback, useEffect, useRef, useState } from "react";
import { getMeta, streamChat, type ChatEvent, type Done, type Meta, type Step } from "./api";
import About from "./About";
import DocDrawer from "./DocDrawer";
import Markdown from "./Markdown";
import ThemeToggle from "./ThemeToggle";
import Timeline from "./Timeline";

type Turn = {
  id: number;
  question: string;
  steps: Step[];
  draft: string; // answer text streamed so far, only on routes with no checks after (replaced by done.answer)
  done: Done | null;
  error: string | null;
  running: boolean;
  open: boolean; // timeline expanded
};

const ROUTE_LABEL: Record<string, string> = {
  knowledge_base: "Knowledge base",
  action: "Agent + tools",
  small_talk: "Small talk",
  out_of_scope: "Out of scope",
  distress: "Wellbeing",
};

// what OrbitBot is doing now, from the last step that finished (shown until the checked answer arrives)
function statusFor(last: Step | undefined): string {
  switch (last?.node) {
    case undefined:
    case "start_turn":
      return "Reading your question…";
    case "decide_retrieval":
    case "plan_queries":
    case "rewrite_query":
      return "Searching the documents…";
    case "retrieve":
      return "Checking which sources are relevant…";
    case "grade_documents":
      return "Writing an answer…";
    case "generate":
    case "revise_answer":
      return "Checking the answer against the sources…";
    case "check_support":
      return last.data?.support === "fully" ? "Making sure the answer helps…" : "Improving the answer…";
    case "check_useful":
      return last.data?.useful === false ? "Searching again…" : "Finishing up…";
    case "agent":
    case "tools":
      return "Looking up your details…";
    default:
      return "Working on it…";
  }
}

let nextId = 1;

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [persona, setPersona] = useState<string>("india");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [doc, setDoc] = useState<string | null>(null);
  const [about, setAbout] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const running = turns.some((t) => t.running);

  // the Space may be waking up: keep retrying /meta for a while instead of failing on the first try
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let attempt = 0; attempt < 30 && !cancelled; attempt++) {
        try {
          const m = await getMeta();
          if (!cancelled) {
            setMeta(m);
            setMetaError(null);
          }
          return;
        } catch (e) {
          if (!cancelled) setMetaError((e as Error).message);
          await new Promise((r) => setTimeout(r, 4000));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const update = (id: number, fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t) => (t.id === id ? fn(t) : t)));

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || running) return;
      const id = nextId++;
      setInput("");
      setTurns((ts) => [
        ...ts.map((t) => ({ ...t, open: false })),
        { id, question: message, steps: [], draft: "", done: null, error: null, running: true, open: true },
      ]);
      const controller = new AbortController();
      abortRef.current = controller;

      const onEvent = (e: ChatEvent) => {
        switch (e.type) {
          case "start":
            setThreadId(e.thread_id);
            break;
          case "step":
            update(id, (t) => ({ ...t, steps: [...t.steps, { node: e.node, data: e.data, t: e.t }] }));
            break;
          case "token":
            update(id, (t) => ({ ...t, draft: t.draft + e.text }));
            break;
          case "done":
            update(id, (t) => ({ ...t, done: e, running: false }));
            break;
          case "error":
            update(id, (t) => ({ ...t, error: e.message, running: false }));
            break;
        }
      };

      try {
        await streamChat({ message, persona: persona || null, thread_id: threadId }, onEvent, controller.signal);
        update(id, (t) => (t.running ? { ...t, running: false, error: t.error ?? "The answer was cut off. Please try again." } : t));
      } catch (e) {
        const aborted = (e as Error).name === "AbortError";
        update(id, (t) => ({ ...t, running: false, error: aborted ? "Stopped." : (e as Error).message }));
      } finally {
        abortRef.current = null;
      }
    },
    [running, persona, threadId],
  );

  const newChat = () => {
    abortRef.current?.abort();
    setTurns([]);
    setThreadId(null);
  };

  const changePersona = (p: string) => {
    setPersona(p);
    newChat(); // a conversation belongs to one person
  };

  const maxChars = meta?.max_question_chars ?? 500;

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden>
            <circle cx="16" cy="16" r="6" fill="var(--accent)" />
            <ellipse cx="16" cy="16" rx="14" ry="6" fill="none" stroke="var(--accent)" strokeWidth="2" transform="rotate(-30 16 16)" />
          </svg>
          <div>
            <div className="brand-name">OrbitBot</div>
            <div className="brand-sub">Self-RAG assistant · fictional company</div>
          </div>
        </div>
        <div className="top-actions">
          <label className="persona">
            <span className="sr-only">Signed in as</span>
            <select value={persona} onChange={(e) => changePersona(e.target.value)} disabled={!meta}>
              <option value="">Not signed in</option>
              {meta?.personas.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.name} · {p.location}
                </option>
              ))}
            </select>
          </label>
          <button className="btn ghost" onClick={newChat} disabled={!turns.length}>
            New chat
          </button>
          <button className="btn ghost" onClick={() => setAbout(true)}>
            About
          </button>
        </div>
        <ThemeToggle />
      </header>

      <main className="chat">
        {!turns.length && (
          <section className="welcome">
            <h1>Ask Orbitra's policies anything</h1>
            <p className="lead">
              OrbitBot searches {meta ? meta.documents : "the"} company documents, then <b>checks its own work</b>: are the sources relevant, is every
              claim supported, does the answer help? Open the steps under each answer to watch it reason.
            </p>
            {!meta && (
              <p className="muted small wake">
                {metaError ? "Waking up the demo server, this can take a minute…" : "Connecting…"}
              </p>
            )}
            {meta && (
              <div className="examples">
                {meta.examples.map((q) => (
                  <button key={q} className="chip" onClick={() => send(q)}>
                    {q}
                  </button>
                ))}
              </div>
            )}
          </section>
        )}

        {turns.map((t) => (
          <article key={t.id} className="turn">
            <div className="msg user">{t.question}</div>
            <div className="msg bot">
              <button className="steps-toggle" onClick={() => update(t.id, (x) => ({ ...x, open: !x.open }))} aria-expanded={t.open}>
                <span className={`chev ${t.open ? "open" : ""}`} aria-hidden>
                  ›
                </span>
                {t.running ? "Thinking" : "How I answered"}
                <span className="pills">
                  {t.done?.route && <span className="pill">{ROUTE_LABEL[t.done.route] ?? t.done.route}</span>}
                  {t.done?.support === "fully" && <span className="pill ok">Grounded</span>}
                  {t.done && t.done.unsupported_claims.length > 0 && <span className="pill warn">Unverified parts</span>}
                  {!!t.done?.rewrites && <span className="pill warn">Searched twice</span>}
                  {t.steps.some((s) => s.node === "revise_answer") && <span className="pill warn">Revised</span>}
                  {t.done && <span className="pill muted">{t.done.seconds.toFixed(1)}s</span>}
                  {t.running && <span className="pill muted">{t.steps.length} steps</span>}
                </span>
              </button>
              {t.open && <Timeline steps={t.steps} question={t.question} running={t.running} />}

              {t.done ? (
                <Markdown text={t.done.answer} citations={t.done.citations} onCite={setDoc} />
              ) : t.draft ? (
                <div className="draft">
                  <Markdown text={t.draft} />
                </div>
              ) : t.running ? (
                <p className="status shimmer" role="status">
                  {statusFor(t.steps[t.steps.length - 1])}
                </p>
              ) : null}
              {t.error && <p className="error">{t.error}</p>}
            </div>
          </article>
        ))}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value.slice(0, maxChars))}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder={meta ? "Ask about leave, expenses, VPN, deployments…" : "Waiting for the server…"}
            rows={1}
            disabled={!meta}
            aria-label="Your question"
          />
          {running ? (
            <button type="button" className="btn" onClick={() => abortRef.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" className="btn primary" disabled={!meta || !input.trim()}>
              Send
            </button>
          )}
        </form>
        <div className="fine muted small">
          {input.length > maxChars * 0.8 && `${input.length}/${maxChars} · `}
          Demo with synthetic data. Answers can take 5–20 s because every one is checked before you see it.
        </div>
      </footer>

      {doc && <DocDrawer docId={doc} onClose={() => setDoc(null)} />}
      {about && <About onClose={() => setAbout(false)} />}
    </div>
  );
}
