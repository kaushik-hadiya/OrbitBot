"""End-to-end pipeline eval with DeepEval: faithfulness + answer relevancy.

Runs the whole graph (main.app) on every question, exactly as a user would hit it:
decide_retrieval -> plan_queries -> retrieve -> grade_documents -> generate -> check_support
(-> revise_answer / flag_unsupported) -> check_useful (-> rewrite_query / suggest_contact).
generator_eval.py scores generate alone on frozen context; this scores the final answer the user
sees, so comparing the two shows what the self-correction loop (support check, revision,
rewrites) adds or costs.

    input             = the original question
    actual_output     = the final answer, minus the suffixes the graph appends from templates
                        (flag_unsupported's "couldn't verify" list, suggest_contact's contact tip):
                        they aren't generated text, and the tip's owner email isn't in any chunk,
                        so the faithfulness judge would wrongly count them as unsupported claims
    retrieval_context = the final state's "documents": the chunks grade_documents kept on the
                        LAST retrieval pass (after any rewrite), as "[doc_id: title] text" like
                        main.format_docs gives them. That's what generate / revise_answer / check_support
                        saw, so faithfulness measures grounding in what the model was actually given.

How each route is scored:
    answered from context (generate, maybe revise / flag)  -> faithfulness + answer relevancy
    generate_direct (decider chose small talk)             -> answer relevancy only (no context)
    decline_out_of_scope / support_wellbeing (canned)      -> no generated answer: relevancy 0,
                                                              faithfulness N/A, nothing sent to the judge
    suggest_contact with no kept chunks (canned reply)     -> no generated answer: relevancy 0,
                                                              faithfulness N/A, nothing sent to the judge
    unanswerable questions (no gold_docs)                  -> no DeepEval; a phrase check reports
                                                              whether the bot said it doesn't know

Results go to evalres/pipeline_<timestamp>.json with the full answers, the scored text and the
context, so --answers can re-score a run without running the graph again.

Run from the project root:
    uv run python eval/pipeline_eval.py
    uv run python eval/pipeline_eval.py --limit 3
    uv run python eval/pipeline_eval.py --qids Q014 Q038
    uv run python eval/pipeline_eval.py --answers evalres/pipeline_<timestamp>.json   # re-score only
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                  # main.py opens documents/, chroma_db/, models/ relative to the root
sys.path.insert(0, str(ROOT))
# DeepEval's per-task timeout also runs while a task waits for a concurrency slot; give it room
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "900")

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric  # noqa: E402
from deepeval.models import OpenRouterModel  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import main  # noqa: E402  (builds/loads the vectorstore, BM25 and reranker)
from retrieval_eval import QUESTIONS, gold_doc_recall, score, with_retries  # noqa: E402

RESULTS_DIR = ROOT / "evalres"
METRICS = ("Faithfulness", "Answer Relevancy")

# text the graph appends from templates (must match flag_unsupported / suggest_contact in main.py)
FLAG_SUFFIX = "\n\n⚠️ I couldn't verify these parts against the knowledge base:\n"
CONTACT_SUFFIXES = ("\n\nThe closest policy is ", "\n\nTry rephrasing, or raise an OrbitDesk ticket")
NO_ANSWER_PREFIX = "I couldn't find an answer to this in the Orbitra knowledge base. "
# cheap check for unanswerable questions: did the bot say it doesn't know?
ABSTAIN_PHRASES = ("don't know", "do not know", "couldn't find", "could not find", "couldn't locate",
                   "could not locate", "unable to find", "i don't have", "i do not have",
                   "not in the knowledge base",
                   "no information", "not available", "does not contain", "doesn't contain",
                   "not mentioned", "not specified", "isn't covered", "is not covered", "not covered")


def load_questions(qids: list[str] | None, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if qids:
        rows = [r for r in rows if r["qid"] in set(qids)]
    return rows[:limit] if limit else rows


def run_graph(question: str) -> dict:
    """Stream the graph, recording which nodes ran; the merged updates are the final state
    (GraphState has no reducers, so each update simply overwrites its keys)."""
    state, route = {"question": question}, []
    started = time.perf_counter()
    for step in main.app.stream({"question": question}, stream_mode="updates"):
        for node, update in step.items():
            route.append(node)
            state |= update or {}
    return {"state": state, "route": route, "seconds": round(time.perf_counter() - started, 1)}


def generated_part(answer: str, route: list[str]) -> str | None:
    """The LLM-written part of the final answer, or None when the answer is entirely a template."""
    if "decline_out_of_scope" in route or "support_wellbeing" in route:
        return None
    if "suggest_contact" in route:
        if answer.startswith(NO_ANSWER_PREFIX):
            return None
        cut = max(answer.rfind(s) for s in CONTACT_SUFFIXES)
        answer = answer[:cut] if cut != -1 else answer
    if "flag_unsupported" in route:
        cut = answer.rfind(FLAG_SUFFIX)
        answer = answer[:cut] if cut != -1 else answer
    return answer.strip() or None


def outcome(route: list[str], has_answer: bool) -> str:
    """One label per way the graph can end, e.g. 'revised -> useful' or 'rewrite + contact (no answer)'."""
    if "decline_out_of_scope" in route:
        label = "declined (out of scope)"
    elif "support_wellbeing" in route:
        label = "wellbeing support"
    elif "generate_direct" in route:
        label = "direct (no retrieval)"
    elif "agent" in route:
        label = "agent (tools)"
    elif "suggest_contact" in route:
        label = "contact + answer" if has_answer else "contact (no answer)"
    elif "flag_unsupported" in route:
        label = "flagged unsupported"
    elif "revise_answer" in route:
        label = "revised -> useful"
    else:
        label = "answered -> useful"
    return f"rewrite + {label}" if "rewrite_query" in route else label


def abstained(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return any(p in text for p in ABSTAIN_PHRASES)


def to_record(q: dict, run: dict) -> dict:
    """Everything needed to score (and re-score) one question, JSON-ready."""
    st, route = run["state"], run["route"]
    docs = st.get("documents") or []
    scored = generated_part(st.get("answer", ""), route)
    rec = {
        "qid": q["qid"],
        "type": q.get("type"),
        "difficulty": q.get("difficulty"),
        "question": q["question"],
        "route": route,
        "outcome": outcome(route, scored is not None),
        "seconds": run["seconds"],
        "sub_questions": st.get("sub_questions"),
        "rewritten_query": st.get("search_query"),
        "generation_attempts": st.get("generation_attempts"),
        "support": st.get("support"),
        "unsupported_claims": st.get("unsupported_claims"),
        "is_useful": st.get("is_useful"),
        "answer": st.get("answer", ""),          # what the user sees
        "scored_answer": scored,                 # what the judge sees (template suffixes stripped)
        "context": [{"doc_id": d.metadata.get("doc_id", "?"), "title": d.metadata.get("title", "?"),
                     "status": d.metadata.get("status"), "text": d.page_content} for d in docs],
    }
    if q.get("gold_docs"):
        rec["gold_docs"] = q["gold_docs"]
        rec["gold_doc_recall"] = gold_doc_recall(docs, q["gold_docs"]) if docs else 0.0
    return rec


def metric(score_=None, reason=None, error=None, na=False) -> dict:
    return {"score": score_, "reason": reason, "error": error, "na": na}


def main_eval(args):
    questions = load_questions(args.qids, args.limit)
    if not questions:
        sys.exit("No questions selected.")

    # 1) the graph: reuse a saved run, or run it (sequentially: the local reranker is shared)
    records, failed = {}, {}
    if args.answers:
        saved = {r["qid"]: r for r in json.loads(Path(args.answers).read_text(encoding="utf-8"))["results"]}
        records = {q["qid"]: saved[q["qid"]] for q in questions if q["qid"] in saved}
        print(f"Reusing {len(records)} pipeline runs from {args.answers}")
    else:
        print(f"Running the full graph for {len(questions)} questions...")
        for i, q in enumerate(questions, 1):
            try:
                rec = to_record(q, with_retries(lambda: run_graph(q["question"])))
            except Exception as e:      # keep going; the question is reported as failed
                failed[q["qid"]] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  [{i}/{len(questions)}] {q['qid']}: FAILED ({type(e).__name__})")
                continue
            records[q["qid"]] = rec
            print(f"  [{i}/{len(questions)}] {q['qid']}: {rec['outcome']}, {len(rec['context'])} chunks, "
                  f"{rec['seconds']}s")
    questions = [q for q in questions if q["qid"] in records]
    answerable = [q for q in questions if q.get("gold_docs")]

    # 2) DeepEval on answerable questions; cases without context get answer relevancy only
    with_ctx, no_ctx = [], []
    for q in answerable:
        rec = records[q["qid"]]
        if rec["scored_answer"] is None:
            continue                     # canned "couldn't find" reply: scored 0 below, no judge call
        tc = LLMTestCase(name=q["qid"], input=q["question"], actual_output=rec["scored_answer"],
                         retrieval_context=[main.format_docs([Document(c["text"], metadata=c)])
                                            for c in rec["context"]] or None)
        (with_ctx if rec["context"] else no_ctx).append(tc)

    judge = OpenRouterModel(model=args.judge, api_key=os.environ["OPENROUTER_API_KEY"], temperature=0)
    started = time.time()
    by_name = {}
    if with_ctx:
        print(f"\nScoring {len(with_ctx)} answers (faithfulness + answer relevancy) with judge {args.judge}...")
        by_name |= score(with_ctx, [FaithfulnessMetric(model=judge), AnswerRelevancyMetric(model=judge)],
                         args.concurrency)
    if no_ctx:
        print(f"\nScoring {len(no_ctx)} answers without context (answer relevancy only)...")
        by_name |= score(no_ctx, [AnswerRelevancyMetric(model=judge)], args.concurrency)

    # 3) one row per question
    results = []
    for q in questions:
        rec = dict(records[q["qid"]])
        if not q.get("gold_docs"):
            rec["abstained"] = abstained(rec["answer"])
        elif rec["scored_answer"] is None:
            rec["metrics"] = {"Faithfulness": metric(na=True, reason="no generated answer"),
                              "Answer Relevancy": metric(0.0, reason="no generated answer (suggest_contact)")}
        else:
            ms = {m: metric(error="not scored") for m in METRICS}
            if not rec["context"]:
                ms["Faithfulness"] = metric(na=True, reason="no retrieval context")
            for md in (by_name[q["qid"]].metrics_data or []) if q["qid"] in by_name else []:
                ms[md.name] = metric(md.score, md.reason, md.error)
            rec["metrics"] = ms
        results.append(rec)
    summary = summarize(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "stage": "pipeline",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "judge": args.judge,
            "generator_llm": main.OPENROUTER["model"],
            "top_k": main.TOP_K, "top_k_per_part": main.TOP_K_PER_PART,
            "max_sub_questions": main.MAX_SUB_QUESTIONS, "candidates": main.CANDIDATES,
            "max_generations": main.MAX_GENERATIONS, "max_rewrites": main.MAX_REWRITES,
            "answers_from": args.answers,
            "scoring_seconds": round(time.time() - started, 1),
        },
        "summary": summary,
        "failures": failed,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary, results)
    if failed:
        print(f"\nPipeline failed for {len(failed)} question(s), not scored: {', '.join(failed)}")
    print(f"\nSaved to {out.relative_to(ROOT)}")


def summarize(results: list[dict]) -> dict:
    def stats(group: list[dict]) -> dict:
        out = {"n": len(group)}
        for m in METRICS:
            blocks = [r["metrics"][m] for r in group]
            scored = [b["score"] for b in blocks if b["score"] is not None]
            out[m] = round(mean(scored), 3) if scored else None
            out[f"{m} n/a"] = sum(b["na"] for b in blocks)
            out[f"{m} errors"] = sum(b["score"] is None and not b["na"] for b in blocks)
        recalls = [r["gold_doc_recall"] for r in group]
        out["gold_doc_recall"] = round(mean(recalls), 3) if recalls else None
        out["avg_seconds"] = round(mean(r["seconds"] for r in group), 1) if group else None
        return out

    answerable = [r for r in results if "metrics" in r]
    unanswerable = [r for r in results if "abstained" in r]
    by_type, by_outcome = defaultdict(list), defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r)
        by_outcome[r["outcome"]].append(r)
    return {
        "overall": stats(answerable),
        "by_type": {t: stats(g) for t, g in sorted(by_type.items())},
        "by_outcome": {o: stats(g) for o, g in sorted(by_outcome.items(), key=lambda kv: -len(kv[1]))},
        "unanswerable": {"n": len(unanswerable), "abstained": sum(r["abstained"] for r in unanswerable),
                         "answered_anyway": [r["qid"] for r in unanswerable if not r["abstained"]]},
    }


def print_summary(summary: dict, results: list[dict], low: float = 0.7):
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    for title, groups in (("BY TYPE", summary["by_type"]), ("BY OUTCOME", summary["by_outcome"])):
        print(f"\n{title:30} {'n':>3}  {'faithfulness':>12}  {'answer relevancy':>16}  "
              f"{'gold-doc recall':>15}  {'avg s':>6}")
        for name, s in [("OVERALL", summary["overall"]), *groups.items()]:
            print(f"{name:30} {s['n']:>3}  {fmt(s['Faithfulness']):>12}  {fmt(s['Answer Relevancy']):>16}  "
                  f"{fmt(s['gold_doc_recall']):>15}  {s['avg_seconds'] or 0:>6.1f}")
    o = summary["overall"]
    if o["Faithfulness n/a"]:
        print(f"Faithfulness n/a (no context) for {o['Faithfulness n/a']} answer(s).")
    errors = sum(o[f"{m} errors"] for m in METRICS)
    if errors:
        print(f"{errors} metric score(s) failed; see 'error' in the JSON.")
    for m in METRICS:
        weak = [f"{r['qid']} ({r['metrics'][m]['score']:.2f})" for r in results
                if "metrics" in r and r["metrics"][m]["score"] is not None and r["metrics"][m]["score"] < low]
        print(f"{m} < {low}: {', '.join(weak) or 'none'}")
    u = summary["unanswerable"]
    if u["n"]:
        print(f"\nUNANSWERABLE: said it doesn't know on {u['abstained']}/{u['n']} "
              f"(answered anyway: {', '.join(u['answered_anyway']) or 'none'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", help="saved pipeline_*.json: re-score its answers, don't run the graph")
    parser.add_argument("--qids", nargs="+", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--judge", default="openai/gpt-oss-120b", help="OpenRouter model for DeepEval's judge")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel judge calls")
    main_eval(parser.parse_args())
