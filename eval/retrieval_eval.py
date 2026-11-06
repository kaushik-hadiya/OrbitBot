"""Retriever eval with DeepEval: contextual recall + contextual precision.

--stage before (default): runs  plan_queries -> retrieve
    and scores the chunks retrieve returns (round-robin merged, in that order).
--stage after:            runs  plan_queries -> retrieve -> grade_documents
    and scores BOTH lists from that same retrieval: the chunks before grading and the ones the
    grader kept. The planner isn't deterministic, so scoring both from one retrieval is what makes
    the before/after comparison fair (otherwise planner noise mixes into the grader's effect).

decide_retrieval is skipped on purpose: we want to score the retriever on every question,
not only on the ones the decider sends to it. Unanswerable questions (no gold_docs) are
skipped, since there is nothing to recall.

    input             = the original question (the judge rates relevance against it)
    expected_output   = ground_truth (recall checks each of its statements is in the chunks)
    retrieval_context = the chunks, as "[doc_id: title] text" (main.format_docs), like generate sees them

If the grader drops every chunk, the graph would go to rewrite_query; here that question's
after-grading context is empty and scores 0 on both metrics (DeepEval can't score an empty list).

Results go to evalres/before_grading_<timestamp>.json (--stage before) or
evalres/before_vs_after_grading_<timestamp>.json (--stage after: the file holds both score sets).

Run from the project root:
    uv run python eval/retrieval_eval.py                       # before grading, all questions
    uv run python eval/retrieval_eval.py --stage after         # before vs after grading
    uv run python eval/retrieval_eval.py --stage after --limit 3
    uv run python eval/retrieval_eval.py --qids Q014 Q020
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
# DeepEval's per-task timeout also runs while a task waits for a concurrency slot, so with ~190
# metric calls and 5 slots the tail of the queue timed out; give each task more room
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "900")

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig  # noqa: E402
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric  # noqa: E402
from deepeval.models import OpenRouterModel  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

import main  # noqa: E402  (builds/loads the vectorstore, BM25 and reranker)

QUESTIONS = ROOT / "eval" / "eval_questions.jsonl"
RESULTS_DIR = ROOT / "evalres"
METRICS = ("Contextual Recall", "Contextual Precision")


def load_questions(qids: list[str] | None, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("gold_docs")]          # unanswerable: nothing to retrieve
    if qids:
        rows = [r for r in rows if r["qid"] in set(qids)]
    return rows[:limit] if limit else rows


def with_retries(fn, attempts: int = 3):
    """OpenRouter sometimes returns an empty message (finish_reason 'error') to a structured-output
    call, which main.py's retry (BadRequestError only) doesn't catch, so retry the whole step."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                raise
            print(f"    retry {attempt}/{attempts - 1} after {type(e).__name__}")
            time.sleep(2 * attempt)


def run_pipeline(question: str, grade: bool) -> dict:
    """plan_queries -> retrieve (-> grade_documents), as the graph runs them on the first pass."""
    state = {"question": question}
    state |= with_retries(lambda: main.plan_queries(state))
    state |= main.retrieve(state)
    out = {"sub_questions": state["sub_questions"], "docs": state["documents"]}
    if grade:
        graded = with_retries(lambda: main.grade_documents(state))
        out |= {"kept": graded["documents"], "grades": graded["doc_grades"]}
    return out


def doc_id(d) -> str:
    return d.metadata.get("doc_id", "?")


def gold_doc_recall(docs, gold_docs) -> float:
    gold = set(gold_docs)
    return len(gold & {doc_id(d) for d in docs}) / len(gold)


def test_case(name: str, q: dict, docs) -> LLMTestCase:
    return LLMTestCase(
        name=name,
        input=q["question"],
        expected_output=q["ground_truth"],
        retrieval_context=[main.format_docs([d]) for d in docs],
    )


def scores_for(name: str, by_name: dict, empty: bool = False) -> dict:
    if empty:
        return {m: {"score": 0.0, "reason": "grader dropped every chunk", "error": None} for m in METRICS}
    scores = {m: {"score": None, "reason": None, "error": None} for m in METRICS}
    for md in (by_name[name].metrics_data or []) if name in by_name else []:
        scores[md.name] = {"score": md.score, "reason": md.reason, "error": md.error}
    return scores


def score(test_cases: list[LLMTestCase], metrics, concurrency: int, extra_rounds: int = 2) -> dict:
    """Run DeepEval, then re-run any test case that came back without every metric scored
    (timeouts, cancelled tasks, judge errors), up to extra_rounds more times."""
    def complete(tr) -> bool:
        return bool(tr.metrics_data) and len(tr.metrics_data) == len(metrics) and \
            all(md.score is not None for md in tr.metrics_data)

    by_name, todo = {}, test_cases
    for round_ in range(1 + extra_rounds):
        if round_:
            print(f"Re-scoring {len(todo)} test case(s) with missing scores (round {round_}/{extra_rounds})...")
        result = evaluate(
            todo,
            metrics,
            async_config=AsyncConfig(max_concurrent=concurrency),
            display_config=DisplayConfig(print_results=False),
            error_config=ErrorConfig(ignore_errors=True),   # one judge failure shouldn't lose the run
        )
        for tr in result.test_results:
            if complete(tr) or tr.name not in by_name:   # keep a partial result rather than none
                by_name[tr.name] = tr
        todo = [tc for tc in todo if tc.name not in by_name or not complete(by_name[tc.name])]
        if not todo:
            break
    return by_name


def main_eval(args):
    after = args.stage == "after"
    questions = load_questions(args.qids, args.limit)
    if not questions:
        sys.exit("No questions selected.")

    # 1) pipeline: sequential, so the LLM calls and the local reranker don't fight
    print(f"Running {'retrieve + grade_documents' if after else 'retrieve'} for {len(questions)} questions...")
    runs, failed = {}, {}
    for i, q in enumerate(questions, 1):
        try:
            run = run_pipeline(q["question"], grade=after)
        except Exception as e:      # keep going; the question is reported as failed
            failed[q["qid"]] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [{i}/{len(questions)}] {q['qid']}: FAILED ({type(e).__name__})")
            continue
        runs[q["qid"]] = run
        line = (f"  [{i}/{len(questions)}] {q['qid']}: {len(run['sub_questions'])} part(s), "
                f"{len(run['docs'])} chunks from {sorted({doc_id(d) for d in run['docs']})}")
        if after:
            line += f" -> kept {len(run['kept'])}"
        print(line)
    questions_all, questions = questions, [q for q in questions if q["qid"] in runs]

    # 2) DeepEval: test cases named "<qid>|before" / "<qid>|after" so the results map back
    test_cases = [test_case(f"{q['qid']}|before", q, runs[q["qid"]]["docs"]) for q in questions]
    if after:   # an empty context can't be scored; scores_for() gives those 0
        test_cases += [test_case(f"{q['qid']}|after", q, runs[q["qid"]]["kept"])
                       for q in questions if runs[q["qid"]]["kept"]]
    judge = OpenRouterModel(model=args.judge, api_key=os.environ["OPENROUTER_API_KEY"], temperature=0)
    metrics = [ContextualRecallMetric(model=judge), ContextualPrecisionMetric(model=judge)]

    print(f"\nScoring {len(test_cases)} test cases with judge {args.judge}...")
    started = time.time()
    by_name = score(test_cases, metrics, args.concurrency)

    # 3) one row per question: DeepEval scores + a cheap check against gold_docs
    rows = []
    for q in questions:
        run = runs[q["qid"]]
        row = {
            "qid": q["qid"],
            "type": q.get("type"),
            "difficulty": q.get("difficulty"),
            "question": q["question"],
            "sub_questions": run["sub_questions"],
            "gold_docs": q["gold_docs"],
            "retrieved_doc_ids": [doc_id(d) for d in run["docs"]],   # in rank order, may repeat a doc
            "gold_doc_recall": gold_doc_recall(run["docs"], q["gold_docs"]),
            "metrics": scores_for(f"{q['qid']}|before", by_name),
            "chunks": [{"doc_id": doc_id(d), "title": d.metadata.get("title"), "source": d.metadata.get("source"),
                        "status": d.metadata.get("status"), "text": d.page_content} for d in run["docs"]],
        }
        if after:
            kept = run["kept"]
            row["after_grading"] = {
                "kept_doc_ids": [doc_id(d) for d in kept],
                "gold_doc_recall": gold_doc_recall(kept, q["gold_docs"]),
                "metrics": scores_for(f"{q['qid']}|after", by_name, empty=not kept),
                # one grade per retrieved chunk, same order as "chunks"
                "grades": [{"doc_id": g["doc_id"], "relevant": g["relevant"], "reason": g["reason"]}
                           for g in run["grades"]],
            }
        rows.append(row)

    summary = {"before_grading": summarize(rows, lambda r: r)}
    if after:
        summary["after_grading"] = summarize(rows, lambda r: r["after_grading"])
        summary["grader"] = grader_stats(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    name = "before_vs_after_grading" if after else "before_grading"
    out = RESULTS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "stage": name,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "judge": args.judge,
            "generator_llm": main.OPENROUTER["model"],
            "top_k": main.TOP_K, "top_k_per_part": main.TOP_K_PER_PART,
            "max_sub_questions": main.MAX_SUB_QUESTIONS, "candidates": main.CANDIDATES,
            "scoring_seconds": round(time.time() - started, 1),
        },
        "summary": summary,
        "failures": failed,
        "results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary)
    if failed:
        print(f"\nPipeline failed for {len(failed)}/{len(questions_all)} question(s), not scored: "
              f"{', '.join(failed)}")
    print(f"\nSaved to {out.relative_to(ROOT)}")


def summarize(rows: list[dict], part) -> dict:
    """part(row) picks the before- or after-grading block of a row (same keys in both)."""
    def stats(group: list[dict]) -> dict:
        blocks = [part(r) for r in group]
        out = {"n": len(group), "gold_doc_recall": round(mean(b["gold_doc_recall"] for b in blocks), 3)}
        for m in METRICS:
            scored = [b["metrics"][m]["score"] for b in blocks if b["metrics"][m]["score"] is not None]
            out[m] = round(mean(scored), 3) if scored else None
            out[f"{m} errors"] = len(group) - len(scored)
        return out

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"overall": stats(rows), "by_type": {t: stats(g) for t, g in sorted(by_type.items())}}


def grader_stats(rows: list[dict]) -> dict:
    before = sum(len(r["retrieved_doc_ids"]) for r in rows)
    kept = sum(len(r["after_grading"]["kept_doc_ids"]) for r in rows)
    # chunks from a gold doc that the grader threw away (not always wrong: a gold doc has many chunks)
    gold_dropped = sum(1 for r in rows for g in r["after_grading"]["grades"]
                       if not g["relevant"] and g["doc_id"] in r["gold_docs"])
    return {
        "chunks_before": before,
        "chunks_kept": kept,
        "keep_rate": round(kept / before, 3) if before else None,
        "gold_doc_chunks_dropped": gold_dropped,
        "questions_all_dropped": [r["qid"] for r in rows if not r["after_grading"]["kept_doc_ids"]],
        "questions_lost_a_gold_doc": [r["qid"] for r in rows
                                      if r["after_grading"]["gold_doc_recall"] < r["gold_doc_recall"]],
    }


def print_summary(summary: dict):
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    for stage in ("before_grading", "after_grading"):
        if stage not in summary:
            continue
        s_all = summary[stage]
        print(f"\n{stage.upper().replace('_', ' ')}")
        print(f"{'':18} {'n':>3}  {'ctx recall':>10}  {'ctx precision':>13}  {'gold-doc recall':>15}")
        for name, s in [("OVERALL", s_all["overall"]), *s_all["by_type"].items()]:
            print(f"{name:18} {s['n']:>3}  {fmt(s['Contextual Recall']):>10}  "
                  f"{fmt(s['Contextual Precision']):>13}  {fmt(s['gold_doc_recall']):>15}")
        errors = sum(s_all["overall"][f"{m} errors"] for m in METRICS)
        if errors:
            print(f"{errors} metric score(s) failed; see 'error' in the JSON.")
    if "grader" in summary:
        g = summary["grader"]
        print(f"\nGRADER: kept {g['chunks_kept']}/{g['chunks_before']} chunks ({fmt(g['keep_rate'])}), "
              f"dropped {g['gold_doc_chunks_dropped']} chunk(s) from gold docs")
        print(f"  dropped everything: {g['questions_all_dropped'] or 'none'}")
        print(f"  lost a whole gold doc: {g['questions_lost_a_gold_doc'] or 'none'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["before", "after"], default="before",
                        help="before: score retrieve's chunks; after: score them and grade_documents' kept chunks")
    parser.add_argument("--qids", nargs="+", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--judge", default="openai/gpt-oss-120b", help="OpenRouter model for DeepEval's judge")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel judge calls")
    main_eval(parser.parse_args())
