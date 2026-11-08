"""Application-level correctness + completeness eval: DeepEval G-Eval against the gold answers.

pipeline_eval.py asks "is the answer grounded in what was retrieved, and on-topic?". This compares
the answer the user sees to ground_truth in eval_questions.jsonl, so a faithful answer built on the
wrong (or a superseded) chunk still fails:
    Correctness  = is it right? (contradictions, superseded / other-region figures, core fact)
    Completeness = did it leave anything out? (every part of the question, every expected fact;
                   coverage only, a wrong value still counts as addressed)

    input           = the original question
    actual_output   = the final answer minus the template suffixes (pipeline_eval.generated_part),
                      i.e. the same text faithfulness / answer relevancy score
    expected_output = ground_truth (the atomic-fact version of the gold answer; superseded figures
                      are deliberately left out of it, see "notes" in eval_questions.jsonl)
    retrieval_context is NOT shown to the judge: both metrics are about the gold answer, grounding is
    pipeline_eval's faithfulness.

G-Eval with explicit evaluation_steps (no criteria): DeepEval only feeds the steps to the scoring
prompt; criteria would just be used to have the judge write its own steps on every run, which makes
scores drift between runs. A rubric anchors what each 0-10 band means. strict_mode stays off: it
turns the score into pass/fail and DeepEval then skips the logprob weighting; the threshold already
gives a pass rate.

Logprobs: G-Eval's score is the judge's 0-10 score weighted by the probabilities of the score
tokens (the G-Eval paper's method), which DeepEval only does when the response carries logprobs.
LogprobOpenRouterModel sets provider.require_parameters (route only to providers that support
logprobs/top_logprobs) and raises if a response still comes back without them, so DeepEval never
silently falls back to the bare integer; such cases come back as errors and score() re-runs them.

Judge: openai/gpt-4o via OpenRouter (the G-Eval paper's model family). Reasoning judges are no use
here: gpt-oss-120b decides the score while reasoning, so the score token has ~100% probability and
the weighting changes nothing, and its scores still moved by 0.1-0.2 between identical runs. Measured
on Q002/Q014, 2 runs each: gpt-4o spreads probability over neighbouring scores (e.g. 8: 0.61,
9: 0.29, 7: 0.08) and repeats to within 0.01. qwen3-235b-2507, llama-3.3-70b and llama-4-maverick
also put ~100% on one score; gpt-4o-mini and mistral-small-3.2 spread but mini was harsh.

How each case is scored:
    answer generated (from context or generate_direct)   -> G-Eval correctness + completeness
    suggest_contact canned reply (no generated answer)   -> 0.0 on both, no judge call
    unanswerable questions (no gold_docs)                -> no G-Eval; phrase check: did it abstain?

Run from the project root:
    uv run python eval/application_eval.py --answers evalres/pipeline_<timestamp>.json  # reuse answers
    uv run python eval/application_eval.py                                            # run the graph
    uv run python eval/application_eval.py --qids Q002 Q005 --verbose
    uv run python eval/application_eval.py --answers ... --providers parasail cerebras  # pin providers
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
# pipeline_eval chdirs to the root, puts it on sys.path and imports main (vectorstore, reranker)
from pipeline_eval import abstained, load_questions, run_graph, to_record  # noqa: E402
from retrieval_eval import score, with_retries  # noqa: E402

from deepeval.metrics import GEval  # noqa: E402
from deepeval.metrics.g_eval import Rubric  # noqa: E402
from deepeval.models import OpenRouterModel  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

import main  # noqa: E402  (already imported by pipeline_eval; only for config below)

RESULTS_DIR = ROOT / "evalres"

CORRECTNESS_STEPS = [
    "Break the expected output into its individual facts (numbers, durations, amounts, dates, "
    "eligibility conditions, owners, contacts, process steps).",
    "For each fact, check whether the actual output states it with the same value; paraphrasing is "
    "fine, a different number, unit, date, region or policy version is not.",
    "Heavily penalize any statement in the actual output that contradicts the expected output, such as "
    "a figure from a superseded policy presented as current, or another region's rule (India vs US vs UK).",
    "Mentioning an outdated figure is acceptable only if the actual output clearly labels it as "
    "superseded, old or no longer applicable.",
    "Do not penalize extra details that are consistent with the expected output; penalize them only if "
    "they contradict it or change what the answer means.",
    "Score in proportion to how many expected facts are covered, giving most weight to the fact that "
    "directly answers the question: missing it means a low score even if minor facts are present.",
    "The core fact is whatever the expected output gives as the answer. If the expected output itself "
    "says where or how to get the information (e.g. a system to check or a ticket to raise) rather than "
    "the information, an actual output that says it cannot see it and gives that same route is fully "
    "correct. Give the lowest score only when the actual output says it cannot answer or does not know "
    "while the expected output contains the answer itself.",
]

# what each score band means; the steps say how to judge, the rubric anchors the number. Both go into
# the scoring prompt. DeepEval normalizes by the rubric's range (0-10), so the score stays 0-1.
CORRECTNESS_RUBRIC = [
    Rubric(score_range=(0, 1), expected_outcome=(
        "Wrong or no answer: contradicts the core fact, presents a superseded or other-region figure as "
        "current, or declines although the expected output contains the answer itself (not just a route "
        "to it).")),
    Rubric(score_range=(2, 4), expected_outcome=(
        "Core fact missing or wrong, though some secondary facts are right.")),
    Rubric(score_range=(5, 6), expected_outcome=(
        "Core fact right, but most supporting facts missing, or a minor detail contradicts the expected "
        "output.")),
    Rubric(score_range=(7, 8), expected_outcome=(
        "Core fact right, one or two minor supporting details missing, no contradictions.")),
    Rubric(score_range=(9, 10), expected_outcome=(
        "All expected facts present and consistent; extra details do not contradict.")),
]

# Completeness = coverage only: did the answer address every part of the question and every expected
# fact? Whether the stated values are right is correctness's job, so a wrong value still counts as
# "addressed" here and the two metrics separate "left something out" from "got something wrong".
COMPLETENESS_STEPS = [
    "List the parts the input asks for: each separate question, and each condition or case it names "
    "(e.g. 'how many days and how do I apply' has two parts).",
    "List the individual facts in the expected output (numbers, durations, dates, conditions, owners, "
    "contacts, process steps).",
    "For each part of the input and each expected fact, check whether the actual output addresses it; "
    "paraphrasing counts. Judge only whether it is addressed, not whether the stated value is right: "
    "accuracy is scored by a separate metric.",
    "Weigh the parts of the input most: leaving a part of the question unanswered is a major gap, "
    "missing a supporting fact from the expected output is a minor one.",
    "Extra information beyond the expected output neither adds to nor reduces completeness.",
    "If the expected output says where or how to get the information (e.g. a system to check or a "
    "ticket to raise) rather than the information itself, giving that same route is complete.",
    "If the actual output says it cannot answer or does not know while the expected output contains "
    "the answer itself, give the lowest score.",
]

COMPLETENESS_RUBRIC = [
    Rubric(score_range=(0, 1), expected_outcome=(
        "Addresses none of what the input asks, or declines although the expected output contains the "
        "answer itself.")),
    Rubric(score_range=(2, 4), expected_outcome=(
        "At least one part of the question is left unanswered (missing supporting facts alone never put "
        "an answer in this band).")),
    Rubric(score_range=(5, 6), expected_outcome=(
        "Every part of the question is addressed, but most of the supporting facts are missing.")),
    Rubric(score_range=(7, 8), expected_outcome=(
        "Every part of the question is addressed; some supporting facts are missing.")),
    Rubric(score_range=(9, 10), expected_outcome=(
        "Every part of the question is addressed and every expected fact is present.")),
]

# metric name -> (evaluation steps, rubric); the name is also the key in each result row (lowercased)
METRICS = {
    "Correctness": (CORRECTNESS_STEPS, CORRECTNESS_RUBRIC),
    "Completeness": (COMPLETENESS_STEPS, COMPLETENESS_RUBRIC),
}


class LogprobOpenRouterModel(OpenRouterModel):
    """OpenRouterModel whose logprob calls (G-Eval scoring) only go to providers that return logprobs,
    and fail loudly instead of silently losing the probability weighting."""

    def __init__(self, *args, providers: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_prefs = {"require_parameters": True}
        if providers:   # pin providers, e.g. for run-to-run comparability (they serve different quants)
            self.provider_prefs |= {"order": providers, "allow_fallbacks": False}
        self.calls = Counter()   # provider -> raw (logprob) calls

    def _raw_kwargs(self, prompt, top_logprobs, audio) -> dict:
        extra = dict(self.generation_kwargs.get("extra_body") or {})
        extra["provider"] = self.provider_prefs
        return dict(model=self.name, messages=self._build_messages(prompt, audio),
                    temperature=self.temperature, logprobs=True, top_logprobs=top_logprobs,
                    **{k: v for k, v in self.generation_kwargs.items() if k != "extra_body"},
                    extra_body=extra)

    def _check(self, completion):
        provider = (completion.model_extra or {}).get("provider", "?")
        lp = completion.choices[0].logprobs if completion.choices else None
        if not (lp and lp.content and lp.content[0].top_logprobs):
            # not AttributeError: that one makes DeepEval fall back to the unweighted score
            raise RuntimeError(f"no logprobs in the judge response (provider {provider})")
        self.calls[provider] += 1
        return completion, self.calculate_cost(completion.usage.prompt_tokens,
                                               completion.usage.completion_tokens, response=completion)

    def _generate_raw_response(self, prompt, top_logprobs=5, audio=None):
        client = self.load_model(async_mode=False)
        return self._check(client.chat.completions.create(**self._raw_kwargs(prompt, top_logprobs, audio)))

    async def _a_generate_raw_response(self, prompt, top_logprobs=5, audio=None):
        client = self.load_model(async_mode=True)
        return self._check(await client.chat.completions.create(**self._raw_kwargs(prompt, top_logprobs, audio)))


def geval_metric(name: str, judge, threshold: float, verbose: bool) -> GEval:
    steps, rubric = METRICS[name]
    return GEval(
        name=name,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        evaluation_steps=steps,
        rubric=rubric,
        model=judge,
        threshold=threshold,
        verbose_mode=verbose,
    )


def correctness_metric(judge, threshold: float, verbose: bool) -> GEval:
    return geval_metric("Correctness", judge, threshold, verbose)


def completeness_metric(judge, threshold: float, verbose: bool) -> GEval:
    return geval_metric("Completeness", judge, threshold, verbose)


def main_eval(args):
    questions = load_questions(args.qids, args.limit)
    if not questions:
        sys.exit("No questions selected.")

    # 1) answers: reuse a saved pipeline run, or run the graph (sequentially: the reranker is shared)
    records, failed = {}, {}
    if args.answers:
        saved = {r["qid"]: r for r in json.loads(Path(args.answers).read_text(encoding="utf-8"))["results"]}
        records = {q["qid"]: saved[q["qid"]] for q in questions if q["qid"] in saved}
        missing = [q["qid"] for q in questions if q["qid"] not in saved]
        print(f"Reusing {len(records)} pipeline runs from {args.answers}"
              + (f" (not in it: {', '.join(missing)})" if missing else ""))
    else:
        print(f"Running the full graph for {len(questions)} questions...")
        for i, q in enumerate(questions, 1):
            try:
                rec = to_record(q, with_retries(lambda: run_graph(q["question"])))
            except Exception as e:
                failed[q["qid"]] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  [{i}/{len(questions)}] {q['qid']}: FAILED ({type(e).__name__})")
                continue
            records[q["qid"]] = rec
            print(f"  [{i}/{len(questions)}] {q['qid']}: {rec['outcome']}, {rec['seconds']}s")
    questions = [q for q in questions if q["qid"] in records]

    # 2) G-Eval on answerable questions that have a generated answer
    cases = [LLMTestCase(name=q["qid"], input=q["question"], actual_output=records[q["qid"]]["scored_answer"],
                         expected_output=q["ground_truth"])
             for q in questions if q.get("gold_docs") and records[q["qid"]]["scored_answer"] is not None]
    judge = LogprobOpenRouterModel(model=args.judge, api_key=os.environ["OPENROUTER_API_KEY"],
                                   temperature=0, providers=args.providers)
    started = time.time()
    by_name = {}
    if cases:
        print(f"\nScoring {len(cases)} answers (G-Eval {' + '.join(METRICS).lower()}, logprob-weighted) "
              f"with judge {args.judge}...")
        by_name = score(cases, [geval_metric(m, judge, args.threshold, args.verbose) for m in METRICS],
                        args.concurrency)

    # 3) one row per question
    results = []
    for q in questions:
        rec = records[q["qid"]]
        row = {k: rec.get(k) for k in ("qid", "type", "difficulty", "question", "route", "outcome",
                                        "answer", "scored_answer")}
        row["expected_output"] = q.get("ground_truth")
        if not q.get("gold_docs"):
            row["abstained"] = abstained(rec["answer"])
        elif rec["scored_answer"] is None:
            for m in METRICS:
                row[m.lower()] = {"score": 0.0, "reason": "no generated answer (suggest_contact)", "error": None}
        else:
            for m in METRICS:
                row[m.lower()] = {"score": None, "reason": None, "error": "not scored"}
            for md in (by_name[q["qid"]].metrics_data or []) if q["qid"] in by_name else []:
                for m in METRICS:
                    if md.name.startswith(m):   # DeepEval names it "<name> [GEval]"
                        row[m.lower()] = {"score": md.score, "reason": md.reason, "error": md.error}
        if "metrics" in rec:   # scores from the pipeline run, for side-by-side reading
            row["pipeline_metrics"] = {m: v.get("score") for m, v in rec["metrics"].items()}
        results.append(row)
    summary = summarize(results, args.threshold)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"application_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "stage": "application",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "judge": args.judge,
            "judge_providers": args.providers or "any with logprobs (require_parameters)",
            "logprob_calls_by_provider": dict(judge.calls),
            "metrics": {m: {"type": "GEval, evaluation_steps + rubric, params: input/actual_output/expected_output",
                            "evaluation_steps": steps,
                            "rubric": [{"score_range": r.score_range, "expected_outcome": r.expected_outcome}
                                       for r in rubric]}
                        for m, (steps, rubric) in METRICS.items()},
            "threshold": args.threshold,
            "generator_llm": main.OPENROUTER["model"],
            "answers_from": args.answers,
            "scoring_seconds": round(time.time() - started, 1),
        },
        "summary": summary,
        "failures": failed,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary, results, judge.calls, args.threshold)
    if failed:
        print(f"\nPipeline failed for {len(failed)} question(s), not scored: {', '.join(failed)}")
    print(f"\nSaved to {out.relative_to(ROOT)}")


def summarize(results: list[dict], threshold: float) -> dict:
    def stats(group: list[dict]) -> dict:
        out = {"n": len(group)}
        for m in METRICS:
            key = m.lower()
            scored = [r[key]["score"] for r in group if r[key]["score"] is not None]
            out[key] = round(mean(scored), 3) if scored else None
            out[f"{key} pass_rate"] = round(sum(s >= threshold for s in scored) / len(scored), 3) if scored else None
            out[f"{key} errors"] = len(group) - len(scored)
        return out

    answerable = [r for r in results if "correctness" in r]
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


def print_summary(summary: dict, results: list[dict], calls: Counter, threshold: float):
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    keys = [m.lower() for m in METRICS]
    for title, groups in (("BY TYPE", summary["by_type"]), ("BY OUTCOME", summary["by_outcome"])):
        head = "".join(f"  {k:>12}  {'pass':>5}" for k in keys)
        print(f"\n{title:30} {'n':>3}{head}   (pass = share >= {threshold})")
        for name, s in [("OVERALL", summary["overall"]), *groups.items()]:
            cols = "".join(f"  {fmt(s[k]):>12}  {fmt(s[f'{k} pass_rate']):>5}" for k in keys)
            print(f"{name:30} {s['n']:>3}{cols}")
    print()
    for m, k in zip(METRICS, keys):
        weak = sorted((r[k]["score"], r["qid"]) for r in results
                      if k in r and r[k]["score"] is not None and r[k]["score"] < threshold)
        print(f"{m} < {threshold}: {', '.join(f'{q} ({s:.2f})' for s, q in weak) or 'none'}")
    errors = sum(summary["overall"][f"{k} errors"] for k in keys)
    if errors:
        print(f"{errors} metric score(s) missing; see 'error' in the JSON.")
    print(f"Logprob-weighted judge calls by provider: {dict(calls) or 'none'}")
    u = summary["unanswerable"]
    if u["n"]:
        print(f"\nUNANSWERABLE: said it doesn't know on {u['abstained']}/{u['n']} "
              f"(answered anyway: {', '.join(u['answered_anyway']) or 'none'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", help="saved pipeline_*.json: score its answers, don't run the graph")
    parser.add_argument("--qids", nargs="+", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--judge", default="openai/gpt-4o",
                        help="OpenRouter model for the G-Eval judge (needs logprobs, no reasoning)")
    parser.add_argument("--providers", nargs="+",
                        help="pin OpenRouter providers for the judge (slugs, e.g. parasail cerebras); "
                             "default: any provider that returns logprobs")
    parser.add_argument("--threshold", type=float, default=0.7, help="pass threshold for correctness")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel judge calls")
    parser.add_argument("--verbose", action="store_true", help="print G-Eval's steps and reasoning per case")
    main_eval(parser.parse_args())
