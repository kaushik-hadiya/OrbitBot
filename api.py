"""OrbitBot web API: streams each Self-RAG step and the answer to a browser, with guardrails for a public demo.

Run locally:   uv run uvicorn api:app --port 7860
Try it:        curl -N -X POST localhost:7860/chat -H 'content-type: application/json' \
                    -d '{"message": "How long is comp-off valid?", "persona": "india"}'

POST /chat returns Server-Sent Events, one JSON object per `data:` line:
  {"type": "start", "thread_id": ...}            the conversation id: send it back for follow-ups
  {"type": "step",  "node": ..., "data": {...}}  one per graph node, as it finishes (the reflection timeline)
  {"type": "token", "node": ..., "text": ...}    answer text while it's written (a revision starts a new draft)
  {"type": "done",  "answer": ..., ...}          the final answer: it replaces any streamed draft
  {"type": "error", "message": ...}              the turn failed; the conversation so far is kept

Guardrails (a public link spends real money and runs on a small free machine):
- the user picks a demo persona; a free-text employee_id would let anyone read anyone's data
- per-IP rate limit, a daily cap for the whole demo, a question length cap, a few turns at a time
- conversations are forgotten after THREAD_TTL_S idle, and at most MAX_THREADS are kept in memory
- errors are logged here and the user gets a generic message (exception text can leak internals)
The hard cap on cost is the credit limit on the demo's own OpenRouter key; these are the layer above it.
"""
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from datetime import date
from pathlib import Path

# main.py reads documents/, data/, models/ and chroma_db/ by relative path
os.chdir(Path(__file__).resolve().parent)

import anyio  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from langchain_core.messages import AIMessageChunk, RemoveMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import agent_tools  # noqa: E402
import main as orbit  # noqa: E402  (builds the index and loads the models: ~5 s)

log = logging.getLogger("orbitbot.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


MAX_QUESTION_CHARS = env_int("MAX_QUESTION_CHARS", 500)
RATE_LIMIT = env_int("RATE_LIMIT", 10)              # messages per IP ...
RATE_WINDOW_S = env_int("RATE_WINDOW_S", 600)       # ... per 10 minutes
DAILY_LIMIT = env_int("DAILY_LIMIT", 300)           # messages per day for the whole demo
MAX_ACTIVE_TURNS = env_int("MAX_ACTIVE_TURNS", 3)   # turns running at once (the free CPU is small)
THREAD_TTL_S = env_int("THREAD_TTL_S", 3600)        # forget a conversation after 1 h idle
MAX_THREADS = env_int("MAX_THREADS", 500)
TURN_STALE_S = 180                                  # a turn running longer than this is counted as dead
# behind a reverse proxy (Hugging Face Spaces), the client's IP is in X-Forwarded-For; PROXY_HOPS is how many
# proxies append to it. 0 = use the socket peer (local runs). Only the proxy-added entries can be trusted.
PROXY_HOPS = env_int("PROXY_HOPS", 0)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# demo personas: one per region, so the region-specific policies and leave balances show
PERSONAS = {
    "india": "ORB01002",    # Imran Iyer, Bengaluru
    "us": "ORB01005",       # Ava Davis, Austin
    "uk": "ORB01013",       # Oliver Walker, London
}
EXAMPLES = [
    "How long is comp-off valid?",
    "Can I deploy a feature on 9 November?",
    "How many earned leaves do I have left?",
    "My VPN keeps disconnecting, what should I do?",
    "Does Orbitra provide pet insurance?",
    "What's the hotel limit for a trip to Mumbai?",
]
# nodes whose LLM output is the answer the user reads (the others return structured verdicts)
ANSWER_NODES = {"generate", "revise_answer", "generate_direct", "agent"}
CITATION = re.compile(r"\[([A-Z]{2,4}(?:-[A-Z0-9]+)+)\]")
GENERIC_ERROR = "Sorry, something went wrong while answering. Please try again; the conversation so far is kept."

# the documents OrbitBot can cite: only what is in the index (no superseded or team-restricted documents)
DOCS = {d.metadata["doc_id"]: d for d in orbit.load_documents() if d.metadata.get("doc_id")}


# ---------- guardrails ----------
class Limits:
    """In-memory limits. One process, so no shared store is needed; a restart resets them, which is fine."""

    def __init__(self):
        self.lock = threading.Lock()
        self.per_ip: dict[str, deque] = defaultdict(deque)
        self.day, self.day_count = date.today(), 0
        self.active: dict[str, float] = {}               # thread_id -> start time of its running turn
        self.threads: OrderedDict[str, float] = OrderedDict()   # thread_id -> last use, oldest first

    def admit(self, ip: str, thread_id: str) -> None:
        """Reserve a turn, or raise the HTTP error that says why not. release() must follow."""
        now = time.time()
        with self.lock:
            self.active = {t: s for t, s in self.active.items() if now - s < TURN_STALE_S}
            if thread_id in self.active:
                raise HTTPException(409, "This conversation is still answering the previous message.")
            if len(self.active) >= MAX_ACTIVE_TURNS:
                raise HTTPException(503, "OrbitBot is busy right now. Please try again in a few seconds.")
            hits = self.per_ip[ip]
            while hits and now - hits[0] > RATE_WINDOW_S:
                hits.popleft()
            if len(hits) >= RATE_LIMIT:
                raise HTTPException(429, f"Rate limit: {RATE_LIMIT} messages per {RATE_WINDOW_S // 60} minutes. "
                                         "Please wait a little.")
            if date.today() != self.day:
                self.day, self.day_count = date.today(), 0
            if self.day_count >= DAILY_LIMIT:
                raise HTTPException(429, "The demo has reached its daily message limit. Please come back tomorrow.")
            hits.append(now)
            self.day_count += 1
            self.active[thread_id] = now
            self.threads[thread_id] = now
            self.threads.move_to_end(thread_id)
            expired = self._expired(now)
        for t in expired:   # outside the lock: deleting a checkpoint thread is not ours to hold it for
            orbit.chat_app.checkpointer.delete_thread(t)

    def release(self, thread_id: str) -> None:
        with self.lock:
            self.active.pop(thread_id, None)
            self.threads[thread_id] = time.time()

    def _expired(self, now: float) -> list[str]:
        """Forget idle conversations, and the oldest ones beyond MAX_THREADS (never a running one)."""
        out = []
        for t, last in list(self.threads.items()):
            too_many = len(self.threads) > MAX_THREADS
            if t not in self.active and (too_many or now - last > THREAD_TTL_S):
                del self.threads[t]
                out.append(t)
        # ip entries with no recent hits
        for ip in [ip for ip, h in self.per_ip.items() if not h or now - h[-1] > RATE_WINDOW_S]:
            del self.per_ip[ip]
        return out


limits = Limits()


def client_ip(request: Request) -> str:
    if PROXY_HOPS > 0:
        hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        if len(hops) >= PROXY_HOPS:
            return hops[-PROXY_HOPS]
    return request.client.host if request.client else "unknown"


# ---------- what the browser sees of each step ----------
def step_data(node: str, update: dict) -> dict:
    """A small, safe summary of a node's update: no full chunks, prompts or internal state."""
    if node == "start_turn":
        return {"standalone_question": update.get("question")}
    if node == "decide_retrieval":
        return {"route": update.get("route"), "reason": update.get("route_reason"),
                "probability": update.get("route_prob"), "restricted": update.get("restricted")}
    if node == "plan_queries":
        return {"sub_questions": update.get("sub_questions", [])}
    if node == "retrieve":
        return {"chunks": [{"doc_id": d.metadata.get("doc_id"), "title": d.metadata.get("title"),
                            "status": d.metadata.get("status"), "snippet": d.page_content[:160]}
                           for d in update.get("documents", [])]}
    if node == "grade_documents":
        return {"grades": [{"doc_id": g["doc_id"], "title": g.get("title"), "relevant": g["relevant"],
                            "reason": g["reason"]} for g in update.get("doc_grades", [])]}
    if node == "check_support":
        return {"support": update.get("support"), "unsupported_claims": update.get("unsupported_claims", [])}
    if node == "check_useful":
        return {"useful": update.get("is_useful"), "reason": update.get("usefulness_reason")}
    if node == "rewrite_query":
        return {"new_query": update.get("search_query")}
    if node == "agent":
        if "answer" in update:
            return {"final": True}
        return {"tool_calls": [{"name": c["name"], "args": c["args"]}
                               for c in update["messages"][-1].tool_calls]}
    if node == "tools":
        return {"results": [{"name": m.name, "preview": str(m.content)[:300]} for m in update.get("messages", [])]}
    return {}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def rollback(config: dict, before: set[str]) -> None:
    """Remove an unfinished turn's messages (same as main.ask_safely), so the next turn starts clean:
    a dangling user message or tool call would make the LLM API reject every later turn."""
    try:
        added = [RemoveMessage(id=m.id) for m in orbit.chat_app.get_state(config).values.get("messages", [])
                 if m.id not in before]
        if added:
            orbit.chat_app.update_state(config, {"messages": added}, as_node="finish")
    except Exception:
        log.exception("rollback failed for %s", config["configurable"]["thread_id"])


async def stream_turn(message: str, employee_id: str | None, thread_id: str):
    """Drive run_turn from a worker thread, so the sync graph doesn't block the server.
    If the browser disconnects, Starlette cancels this task: the remaining nodes never run, and the finally
    closes run_turn explicitly (left to Starlette, it isn't closed, so its cleanup never ran and the
    conversation stayed locked). The close is shielded because the task is already cancelled by then."""
    turn = run_turn(message, employee_id, thread_id)
    try:
        while (event := await run_in_threadpool(next, turn, None)) is not None:
            yield event
    finally:
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(turn.close)


def run_turn(message: str, employee_id: str | None, thread_id: str):
    """One turn as SSE strings. Its finally rolls back an unfinished turn and frees the conversation."""
    config = {"configurable": {"thread_id": thread_id}}
    completed, before = False, None
    try:
        before = {m.id for m in orbit.chat_app.get_state(config).values.get("messages", [])}
        yield sse({"type": "start", "thread_id": thread_id})
        started = time.time()
        stream = orbit.chat_app.stream({"question": message, "employee_id": employee_id}, config,
                                       stream_mode=["updates", "messages"])
        for mode, payload in stream:
            if mode == "messages":
                chunk, meta = payload
                node = meta.get("langgraph_node")
                if node in ANSWER_NODES and isinstance(chunk, AIMessageChunk) and chunk.content \
                        and not chunk.tool_call_chunks:
                    yield sse({"type": "token", "node": node, "text": chunk.content})
                continue
            for node, update in payload.items():
                if node != "finish" and update is not None:
                    yield sse({"type": "step", "node": node, "data": step_data(node, update),
                               "t": round(time.time() - started, 2)})
        state = orbit.chat_app.get_state(config).values
        answer = state.get("answer", "")
        completed = True
        yield sse({
            "type": "done",
            "thread_id": thread_id,
            "answer": answer,
            "route": state.get("route"),
            "citations": [c for c in dict.fromkeys(CITATION.findall(answer)) if c in DOCS],
            "support": state.get("support") or None,
            "unsupported_claims": state.get("unsupported_claims", []),
            "rewrites": state.get("rewrite_count", 0),
            "seconds": round(time.time() - started, 2),
        })
    except Exception:
        log.exception("turn failed (thread %s)", thread_id)
        yield sse({"type": "error", "message": GENERIC_ERROR})
    finally:   # also runs on a client disconnect (GeneratorExit)
        if not completed and before is not None:
            rollback(config, before)
        limits.release(thread_id)


# ---------- routes ----------
app = FastAPI(title="OrbitBot API", docs_url=None, redoc_url=None)
if ALLOWED_ORIGINS:   # only needed when the frontend is served from another origin (e.g. Vite dev server)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["content-type"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    persona: str | None = None     # a PERSONAS key; None = not signed in
    thread_id: str | None = None   # from the "start" event; None starts a new conversation


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    message = body.message.strip()
    if not message:
        raise HTTPException(422, "The message is empty.")
    if body.persona is not None and body.persona not in PERSONAS:
        raise HTTPException(422, f"Unknown persona. Choose one of: {', '.join(PERSONAS)}.")
    thread_id = body.thread_id or str(uuid.uuid4())
    try:
        uuid.UUID(thread_id)
    except ValueError:
        raise HTTPException(422, "thread_id must be a UUID from a previous reply.")
    limits.admit(client_ip(request), thread_id)
    employee_id = PERSONAS.get(body.persona) if body.persona else None
    return StreamingResponse(stream_turn(message, employee_id, thread_id), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/meta")
def meta():
    """What the UI needs to draw itself: personas, example questions, limits, which checks use Jev."""
    personas = []
    for key, emp_id in PERSONAS.items():
        e = agent_tools.EMPLOYEES_BY_ID[emp_id]
        personas.append({"key": key, "name": e["name"], "title": e["title"], "location": e["location"]})
    return {
        "personas": personas,
        "examples": EXAMPLES,
        "max_question_chars": MAX_QUESTION_CHARS,
        "documents": len(DOCS),
        "decision_model": {"decider": orbit.DECIDER, "grader": orbit.GRADER,
                           "support": orbit.SUPPORT, "useful": orbit.USEFUL},
    }


@app.get("/kb/{doc_id}")
def kb_document(doc_id: str):
    """A cited document, for the citation pop-up. Only indexed documents are served."""
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(404, "No such document.")
    m = doc.metadata
    return {"doc_id": doc_id, "title": m.get("title"), "department": m.get("department"),
            "version": m.get("version"), "status": m.get("status"), "applies_to": m.get("applies_to"),
            "effective_date": m.get("effective_date"), "content": doc.page_content}


@app.get("/health")
def health():
    return {"ok": True}


# the React UI (cd frontend && npm run build), served from the same origin as the API: no CORS needed.
# Mounted last, so the API routes above take precedence over files.
UI_DIR = Path("frontend/dist")
if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
