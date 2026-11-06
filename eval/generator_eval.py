"""Generator eval with DeepEval: faithfulness + answer relevancy.

Scores the raw output of main.generate (prompt | llm), before check_support / revise_answer /
check_useful touch it, so the numbers measure the generator alone.

No retrieval or grading is re-run: the context comes from a saved retrieval eval run
(evalres/before_vs_after_grading_*.json, the latest one by default). For each question it takes
the retrieved chunks the grader kept, i.e. exactly what grade_documents hands to generate.
Only the 47 answerable questions are in that file; unanswerable ones (no gold_docs) aren't scored here.

    input             = the original question
    actual_output     = the raw generate answer
    retrieval_context = the kept chunks, as "[doc_id] text", like generate sees them

Faithfulness checks every claim in the answer against retrieval_context; answer relevancy checks
every statement in the answer addresses the question. Neither needs gold chunks or ground_truth.

Results go to evalres/generator_<timestamp>.json, answers included, so --answers can re-score
them later without paying for the generator again.

Run from the project root:
    uv run python eval/generator_eval.py
    uv run python eval/generator_eval.py --limit 3
    uv run python eval/generator_eval.py --qids Q014 Q020
    uv run python eval/generator_eval.py --answers evalres/generator_<timestamp>.json   # re-score only
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

import main  # noqa: E402
from retrieval_eval import score, with_retries  # noqa: E402  (same DeepEval re-try loop)

RESULTS_DIR = ROOT / "evalres"
METRICS = ("Faithfulness", "Answer Relevancy")


def latest_retrieval_run() -> Path:
    runs = sorted(RESULTS_DIR.glob("before_vs_after_grading_*.json"))
    if not runs:
        sys.exit("No evalres/before_vs_after_grading_*.json found; run retrieval_eval.py --stage after first.")
    return runs[-1]


def kept_chunks(row: dict) -> list[Document]:
    """The chunks grade_documents kept for this question (grades are in the same order as chunks)."""
    return [Document(page_content=c["text"], metadata={"doc_id": c["doc_id"], "source": c["source"],
                                                       "status": c["status"]})
            for c, g in zip(row["chunks"], row["after_grading"]["grades"]) if g["relevant"]]


def generate(row: dict) -> str:
    state = {"question": row["question"], "documents": kept_chunks(row)}
    return with_retries(lambda: main.generate(state))["answer"]


def scores_for(name: str, by_name: dict) -> dict:
    scores = {m: {"score": None, "reason": None, "error": None} for m in METRICS}
    for md in (by_name[name].metrics_data or []) if name in by_name else []:
        scores[md.name] = {"score": md.score, "reason": md.reason, "error": md.error}
    return scores


def main_eval(args):
    source = Path(args.retrieval_run) if args.retrieval_run else latest_retrieval_run()
    rows = json.loads(source.read_text(encoding="utf-8"))["results"]
    if args.qids:
        rows = [r for r in rows if r["qid"] in set(args.qids)]
    rows = rows[:args.limit] if args.limit else rows
    rows = [r for r in rows if any(g["relevant"] for g in r["after_grading"]["grades"])]
    if not rows:
        sys.exit("No questions selected (or the grader kept no chunks for them).")
    print(f"Context from {source.relative_to(ROOT)}")

    # 1) answers: reuse a previous run's, or call the generator (in parallel: no local model involved)
    answers, failed = {}, {}
    if args.answers:
        saved = {r["qid"]: r["answer"] for r in json.loads(Path(args.answers).read_text(encoding="utf-8"))["results"]}
        answers = {r["qid"]: saved[r["qid"]] for r in rows if r["qid"] in saved}
        print(f"Reusing {len(answers)} answers from {args.answers}")
    else:
        print(f"Generating answers for {len(rows)} questions...")
        def run(row):
            try:
                return row["qid"], generate(row), None
            except Exception as e:   # keep going; the question is reported as failed
                return row["qid"], None, f"{type(e).__name__}: {str(e)[:300]}"
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, (qid, answer, err) in enumerate(pool.map(run, rows), 1):
                if err:
                    failed[qid] = err
                    print(f"  [{i}/{len(rows)}] {qid}: FAILED ({err.split(':')[0]})")
                else:
                    answers[qid] = answer
                    print(f"  [{i}/{len(rows)}] {qid}: {len(answer)} chars")
    rows = [r for r in rows if r["qid"] in answers]

    # 2) DeepEval
    test_cases = [LLMTestCase(
        name=r["qid"],
        input=r["question"],
        actual_output=answers[r["qid"]],
        retrieval_context=[f"[{d.metadata['doc_id']}] {d.page_content}" for d in kept_chunks(r)],
    ) for r in rows]
    judge = OpenRouterModel(model=args.judge, api_key=os.environ["OPENROUTER_API_KEY"], temperature=0)
    metrics = [FaithfulnessMetric(model=judge), AnswerRelevancyMetric(model=judge)]

    print(f"\nScoring {len(test_cases)} answers with judge {args.judge}...")
    started = time.time()
    by_name = score(test_cases, metrics, args.concurrency)

    # 3) one row per question
    results = [{
        "qid": r["qid"],
        "type": r["type"],
        "difficulty": r["difficulty"],
        "question": r["question"],
        "context_doc_ids": r["after_grading"]["kept_doc_ids"],
        "answer": answers[r["qid"]],
        "metrics": scores_for(r["qid"], by_name),
    } for r in rows]
    summary = summarize(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"generator_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "stage": "generator",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "judge": args.judge,
            "generator_llm": main.OPENROUTER["model"],
            "context_from": str(source.relative_to(ROOT)),
            "answers_from": args.answers,
            "scoring_seconds": round(time.time() - started, 1),
        },
        "summary": summary,
        "failures": failed,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary, results)
    if failed:
        print(f"\nGeneration failed for {len(failed)} question(s), not scored: {', '.join(failed)}")
    print(f"\nSaved to {out.relative_to(ROOT)}")


def summarize(results: list[dict]) -> dict:
    def stats(group: list[dict]) -> dict:
        out = {"n": len(group)}
        for m in METRICS:
            scored = [r["metrics"][m]["score"] for r in group if r["metrics"][m]["score"] is not None]
            out[m] = round(mean(scored), 3) if scored else None
            out[f"{m} errors"] = len(group) - len(scored)
        return out

    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    return {"overall": stats(results), "by_type": {t: stats(g) for t, g in sorted(by_type.items())}}


def print_summary(summary: dict, results: list[dict], low: float = 0.7):
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    print(f"\n{'':18} {'n':>3}  {'faithfulness':>12}  {'answer relevancy':>16}")
    for name, s in [("OVERALL", summary["overall"]), *summary["by_type"].items()]:
        print(f"{name:18} {s['n']:>3}  {fmt(s['Faithfulness']):>12}  {fmt(s['Answer Relevancy']):>16}")
    errors = sum(summary["overall"][f"{m} errors"] for m in METRICS)
    if errors:
        print(f"{errors} metric score(s) failed; see 'error' in the JSON.")
    for m in METRICS:
        weak = [f"{r['qid']} ({r['metrics'][m]['score']:.2f})" for r in results
                if r["metrics"][m]["score"] is not None and r["metrics"][m]["score"] < low]
        print(f"{m} < {low}: {', '.join(weak) or 'none'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-run", help="saved before_vs_after_grading_*.json to take the context from "
                                                "(default: the latest)")
    parser.add_argument("--answers", help="saved generator_*.json: re-score its answers, don't generate")
    parser.add_argument("--qids", nargs="+", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--judge", default="openai/gpt-oss-120b", help="OpenRouter model for DeepEval's judge")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel generator / judge calls")
    main_eval(parser.parse_args())
