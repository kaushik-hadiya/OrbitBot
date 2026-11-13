"""Client for OpenRouter's Decisions API (alpha): typed answers with probabilities, no generated text.

A Decisions model (default: TypeSafe's Jev) reads `state` and answers named questions:
    choice  pick one option from {name: description}; returns `choice` and `probabilities`
    noul    yes/no; returns the probability of yes
It is not a chat model, so ChatOpenAI can't call it; this is a plain HTTP POST with the same OpenRouter key.

The API is alpha and its shape may change, so every failure (timeout, 4xx, 5xx after retries, an answer
missing from the reply) raises DecisionsUnavailable, and callers fall back to their LLM chain.
"""
import os
import time

import httpx
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

URL = "https://openrouter.ai/api/alpha/decisions"
# pinned rather than "~typesafe/jev-latest", so a new release can't silently change the routing
DECISIONS_MODEL = os.environ.get("DECISIONS_MODEL", "typesafe/jev-1.14")
RETRY_STATUS = {429, 500, 502, 503, 504}
ATTEMPTS = 3

# one client for the whole process: reusing the connection skips a TLS handshake on every call
_client = httpx.Client(timeout=10, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})


class DecisionsUnavailable(Exception):
    """The Decisions API gave no usable answer; the caller should use its LLM chain instead."""


def choice_q(instructions: str, options: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul_q(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true or "", "false": false or ""}
    return q


@traceable(run_type="llm", name="decisions")
def decide(state, questions: dict[str, dict], *, model: str | None = None) -> tuple[dict, dict]:
    """Ask `questions` about `state` in one request. Returns (answers by question id, meta) where meta has
    the served model, latency_s, input_tokens and cost (USD)."""
    body = {"model": model or DECISIONS_MODEL, "state": state, "questions": questions}
    started = time.perf_counter()
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = _client.post(URL, json=body)
        except httpx.HTTPError as e:   # timeout, connection reset, ...
            if attempt == ATTEMPTS:
                raise DecisionsUnavailable(f"{type(e).__name__}: {e}") from e
        else:
            if response.status_code == 200:
                break
            if response.status_code not in RETRY_STATUS or attempt == ATTEMPTS:
                raise DecisionsUnavailable(f"HTTP {response.status_code}: {response.text[:300]}")
        time.sleep(0.5 * 2 ** (attempt - 1))
    try:
        data = response.json()
        answers = {qid: data["answers"][qid] for qid in questions}
    except (ValueError, KeyError, TypeError) as e:
        raise DecisionsUnavailable(f"unexpected reply: {response.text[:300]}") from e
    usage = data.get("usage") or {}
    meta = {"model": data.get("model"), "latency_s": round(time.perf_counter() - started, 3),
            "input_tokens": usage.get("input_tokens"), "cost": usage.get("cost")}
    return answers, meta
