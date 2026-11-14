// Types and calls for the OrbitBot API (api.py). /chat streams Server-Sent Events over a POST,
// which EventSource can't send, so the stream is read from fetch() by hand.

export type Persona = { key: string; name: string; title: string; location: string };

export type Meta = {
  personas: Persona[];
  examples: string[];
  max_question_chars: number;
  documents: number;
  decision_model: Record<string, string>;
};

export type Step = { node: string; data: any; t: number };

export type Done = {
  thread_id: string;
  answer: string;
  route: string | null;
  citations: string[];
  support: string | null;
  unsupported_claims: string[];
  rewrites: number;
  seconds: number;
};

export type ChatEvent =
  | { type: "start"; thread_id: string }
  | ({ type: "step" } & Step)
  | { type: "token"; node: string; text: string }
  | ({ type: "done" } & Done)
  | { type: "error"; message: string };

export type KbDoc = {
  doc_id: string;
  title: string;
  department?: string;
  version?: string | number;
  status?: string;
  applies_to?: string;
  effective_date?: string;
  content: string;
};

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "That message can't be sent (is it too long?).";
  } catch {
    /* not JSON */
  }
  return res.status === 502 || res.status === 503
    ? "OrbitBot is starting up or busy. Please try again in a moment."
    : `Request failed (${res.status}).`;
}

export async function getMeta(): Promise<Meta> {
  const res = await fetch("/meta");
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function getDoc(docId: string): Promise<KbDoc> {
  const res = await fetch(`/kb/${encodeURIComponent(docId)}`);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function streamChat(
  body: { message: string; persona: string | null; thread_id: string | null },
  onEvent: (e: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorText(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data: "))
        .map((l) => l.slice(6))
        .join("\n");
      if (data) onEvent(JSON.parse(data) as ChatEvent);
    }
  }
}
