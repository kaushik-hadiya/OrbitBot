"""Grader before/after: the LLM grader vs the Decisions-model (Jev) grader on the same retrieved chunks.

For each question with gold docs (eval/eval_questions.jsonl), plan_queries + retrieve run once, and the same
chunks are then graded three ways:
    llm          main.grader, one LLM call per chunk in parallel (what grade_documents does today)
    jev_batched  one Decisions request: every chunk in the state, one yes/no question per chunk
    jev_chunk    one small Decisions request per chunk, in parallel

No judge model: chunks are scored against the gold docs. A "gold chunk" is a chunk from one of the question's
gold docs (a proxy: a gold doc's chunk can still be off-topic for the question). Per grader it reports
    keep rate, gold-chunk recall (gold chunks kept), precision (kept chunks that are gold),
    gold-doc recall after grading (as retrieval_eval measures it), questions where every chunk was dropped
    (the graph would rewrite the query), agreement with the LLM per chunk, latency and cost of the step.
Jev graders are replayed at several TAU_REL thresholds from the saved probabilities (no new calls).

The retrieval is saved with the results; --retrieval <file> grades a saved retrieval again without re-running
plan_queries/retrieve, so different graders or thresholds see identical chunks.

Results go to evalres/grader_compare_<timestamp>.json. DeepEval's contextual metrics come afterwards from
    GRADER=jev uv run python eval/retrieval_eval.py --stage after

Run from the project root:
    uv run python eval/grader_compare.py
    uv run python eval/grader_compare.py --qids Q001 Q014 --graders llm jev_batched
    uv run python eval/grader_compare.py --retrieval evalres/grader_compare_<timestamp>.json
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document

from decider_compare import ROOT, llm_price, pct  # noqa: E402  (also chdirs to the root and imports main)

import decisions  # noqa: E402
import main  # noqa: E402

QUESTIONS = ROOT / "eval" / "eval_questions.jsonl"
RESULTS_DIR = ROOT / "evalres"
GRADERS = ["llm", "jev_batched", "jev_chunk"]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_questions(qids, limit) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("gold_docs")]          # unanswerable: nothing to grade against
    if qids:
        rows = [r for r in rows if r["qid"] in set(qids)]
    return rows[:limit] if limit else rows


def retrieve(q: dict) -> dict:
    """plan_queries -> retrieve, as the graph runs them on the first pass (decide_retrieval skipped)."""
    state = {"question": q["question"], "search_question": q["question"]}
    for attempt in range(3):      # OpenRouter's occasional empty structured reply
        try:
            state |= main.plan_queries(state)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    state |= main.retrieve(state)
    return {"sub_questions": state["sub_questions"],
            "chunks": [{"text": d.page_content, "metadata": d.metadata} for d in state["documents"]]}


def grade_llm(question, parts, docs, price) -> dict:
    usage = UsageMetadataCallbackHandler()
    started = time.perf_counter()
    grades = main.grader.batch([
        {"question": question, "parts": "; ".join(parts), "chunk": d.page_content,
         **{k: d.metadata.get(k, "?") for k in main.GRADE_FIELDS}} for d in docs
    ], config={"callbacks": [usage]}, return_exceptions=True)
    seconds = time.perf_counter() - started
    tokens = list(usage.usage_metadata.values())
    inp, out = sum(t.get("input_tokens", 0) for t in tokens), sum(t.get("output_tokens", 0) for t in tokens)
    return {"keep": [True if isinstance(g, Exception) else g.is_relevant for g in grades],   # as main does
            "reasons": [f"error {type(g).__name__}" if isinstance(g, Exception) else g.reason for g in grades],
            "latency_s": round(seconds, 3), "requests": len(docs), "input_tokens": inp, "output_tokens": out,
            "cost": inp * price["prompt"] + out * price["completion"] if price else None}


def grade_jev(question, parts, docs, mode) -> dict:
    try:
        probs, meta = main.jev_grade(question, parts, docs, mode=mode)
    except decisions.DecisionsUnavailable as e:
        return {"error": str(e)[:200]}
    return {"probs": probs, **meta}


def keep_at(g: dict, tau: float) -> list[bool]:
    return g["keep"] if "keep" in g else [p >= tau for p in g["probs"]]


def stats(runs: list[dict], name: str, tau: float) -> dict:
    ok = [r for r in runs if "error" not in r["grades"][name]]
    chunks = gold = kept = kept_gold = all_dropped = lost_gold_doc = agree = 0
    doc_recalls = []
    for r in ok:
        keep = keep_at(r["grades"][name], tau)
        llm_keep = r["grades"]["llm"]["keep"] if "llm" in r["grades"] else None
        golds = [c["metadata"].get("doc_id") in r["gold_docs"] for c in r["chunks"]]
        chunks += len(keep)
        gold += sum(golds)
        kept += sum(keep)
        kept_gold += sum(k and g for k, g in zip(keep, golds))
        all_dropped += bool(keep) and not any(keep)
        before = {c["metadata"].get("doc_id") for c in r["chunks"]} & set(r["gold_docs"])
        after = {c["metadata"].get("doc_id") for c, k in zip(r["chunks"], keep) if k} & set(r["gold_docs"])
        lost_gold_doc += len(before - after) > 0
        doc_recalls.append(len(after) / len(r["gold_docs"]))
        if llm_keep:
            agree += sum(a == b for a, b in zip(keep, llm_keep))
    return {"tau": tau if name != "llm" else None, "questions": len(ok), "chunks": chunks,
            "keep_rate": round(kept / chunks, 3) if chunks else None,
            "gold_chunk_recall": round(kept_gold / gold, 3) if gold else None,
            "precision": round(kept_gold / kept, 3) if kept else None,
            "gold_doc_recall": round(statistics.mean(doc_recalls), 3) if doc_recalls else None,
            "questions_all_dropped": all_dropped, "questions_losing_a_gold_doc": lost_gold_doc,
            "agreement_with_llm": round(agree / chunks, 3) if chunks and name != "llm" else None}


def cost_latency(runs: list[dict], name: str) -> dict:
    rows = [r["grades"][name] for r in runs if "error" not in r["grades"][name]]
    lat = [g["latency_s"] for g in rows]
    costs = [g["cost"] for g in rows if g.get("cost") is not None]
    return {"latency_p50": pct(lat, 0.5), "latency_p95": pct(lat, 0.95),
            "latency_mean": round(statistics.mean(lat), 3) if lat else None,
            "requests_per_question": round(statistics.mean(g["requests"] for g in rows), 2) if rows else None,
            "input_tokens_per_question": round(statistics.mean(g["input_tokens"] for g in rows), 1) if rows else None,
            "cost_per_question": statistics.mean(costs) if costs else None,
            "errors": sum("error" in r["grades"][name] for r in runs)}


def summarize(runs: list[dict], graders: list[str]) -> dict:
    before = {"gold_doc_recall": round(statistics.mean(
        len({c["metadata"].get("doc_id") for c in r["chunks"]} & set(r["gold_docs"])) / len(r["gold_docs"])
        for r in runs), 3), "chunks": sum(len(r["chunks"]) for r in runs)}
    out = {"before_grading": before, "graders": {}}
    for name in graders:
        out["graders"][name] = {"at_tau": stats(runs, name, main.TAU_REL), **cost_latency(runs, name)}
        if name != "llm":
            out["graders"][name]["tau_sweep"] = [stats(runs, name, t) for t in TAUS]
    return out


def print_report(summary: dict, runs: list[dict], graders: list[str]):
    b = summary["before_grading"]
    print(f"\nBefore grading: {b['chunks']} chunks, gold-doc recall {b['gold_doc_recall']}")
    cols = ["keep_rate", "gold_chunk_recall", "precision", "gold_doc_recall", "questions_all_dropped",
            "questions_losing_a_gold_doc", "agreement_with_llm"]
    short = ["keep", "goldchunk", "precision", "golddoc", "all-drop", "lost-gold", "agree"]
    print(f"\nAt TAU_REL={main.TAU_REL} (LLM grader has no threshold):")
    print(f"{'grader':13}" + "".join(f"{s:>10}" for s in short))
    for name in graders:
        s = summary["graders"][name]["at_tau"]
        print(f"{name:13}" + "".join(f"{str(s[c]) if s[c] is not None else '-':>10}" for c in cols))
    print(f"\n{'grader':13}{'p50 s':>8}{'p95 s':>8}{'mean s':>8}{'reqs':>7}{'in tok':>9}{'$ / question':>14}{'errors':>8}")
    for name in graders:
        s = summary["graders"][name]
        cost = f"{s['cost_per_question']:.7f}" if s["cost_per_question"] is not None else "n/a"
        print(f"{name:13}{s['latency_p50']:>8}{s['latency_p95']:>8}{s['latency_mean']:>8}"
              f"{s['requests_per_question']:>7}{s['input_tokens_per_question']:>9}{cost:>14}{s['errors']:>8}")
    for name in graders:
        if name == "llm":
            continue
        print(f"\nThreshold sweep, {name}:")
        print(f"{'tau':>5}" + "".join(f"{s:>10}" for s in short))
        for s in summary["graders"][name]["tau_sweep"]:
            print(f"{s['tau']:>5}" + "".join(f"{str(s[c]) if s[c] is not None else '-':>10}" for c in cols))
    if "llm" in graders:
        for name in [g for g in graders if g != "llm"]:
            diffs = []
            for r in runs:
                g, l = r["grades"][name], r["grades"]["llm"]
                if "error" in g:
                    continue
                for i, (c, a, k) in enumerate(zip(r["chunks"], l["keep"], keep_at(g, main.TAU_REL))):
                    if a != k:
                        diffs.append((r, i, c, a, g["probs"][i]))
            print(f"\nCHUNK DISAGREEMENTS llm vs {name} at TAU_REL={main.TAU_REL} ({len(diffs)}, first 25):")
            for r, i, c, a, p in diffs[:25]:
                gold = "GOLD" if c["metadata"].get("doc_id") in r["gold_docs"] else "    "
                print(f"  {r['qid']} c{i} {gold} {c['metadata'].get('doc_id'):12} llm={'keep' if a else 'drop'} "
                      f"p={p:.2f}  | {r['grades']['llm']['reasons'][i][:90]}")


def main_eval(args):
    questions = load_questions(args.qids, args.limit)
    if not questions:
        sys.exit("No questions selected.")
    saved = {}
    if args.retrieval:
        saved = {r["qid"]: r for r in json.loads(open(args.retrieval, encoding="utf-8").read())["results"]}
        questions = [q for q in questions if q["qid"] in saved]
        print(f"Reusing the retrieval of {len(questions)} questions from {args.retrieval}")

    runs = []
    print(f"Retrieving for {len(questions)} questions (sequential: the local reranker is shared)...")
    for i, q in enumerate(questions, 1):
        if q["qid"] in saved:
            got = {"sub_questions": saved[q["qid"]]["sub_questions"], "chunks": saved[q["qid"]]["chunks"]}
        else:
            got = retrieve(q)
            print(f"  [{i}/{len(questions)}] {q['qid']}: {len(got['chunks'])} chunks")
        runs.append({"qid": q["qid"], "type": q["type"], "question": q["question"], "gold_docs": q["gold_docs"], **got})

    price = llm_price()
    print(f"\nGrading with {args.graders} (Jev = {decisions.DECISIONS_MODEL}) ...")

    def grade(r):
        docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in r["chunks"]]
        r["grades"] = {}
        for name in args.graders:   # one grader after another, so they don't slow each other down
            if name == "llm":
                r["grades"][name] = grade_llm(r["question"], r["sub_questions"], docs, price)
            else:
                r["grades"][name] = grade_jev(r["question"], r["sub_questions"], docs,
                                              "batched" if name == "jev_batched" else "per_chunk")
        print("  " + r["qid"] + "  " + "  ".join(
            f"{n}={g['latency_s']}s" if "error" not in g else f"{n}=ERROR" for n, g in r["grades"].items()))
        return r

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        runs = list(pool.map(grade, runs))

    summary = summarize(runs, args.graders)
    print_report(summary, runs, args.graders)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"grader_compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "stage": "grader_compare", "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"llm": main.OPENROUTER["model"], "llm_effort": "low", "llm_price_per_token": price,
                   "decisions_model": decisions.DECISIONS_MODEL, "tau_rel": main.TAU_REL,
                   "graders": args.graders, "retrieval_from": args.retrieval, "concurrency": args.concurrency},
        "summary": summary, "results": runs,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graders", nargs="+", default=GRADERS, choices=GRADERS)
    parser.add_argument("--retrieval", help="saved grader_compare_*.json: grade its chunks, don't retrieve again")
    parser.add_argument("--qids", nargs="+", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--concurrency", type=int, default=3, help="questions graded in parallel")
    main_eval(parser.parse_args())
