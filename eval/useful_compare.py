"""Usefulness check before/after: the LLM usefulness_checker vs the Decisions-model (Jev) check, same inputs.

Inputs are (question, answer) pairs, nothing is generated:
    real        distinct final answers saved in evalres/pipeline_*.json (knowledge-base route only)
    mismatched  a question paired with the answer to a different question (different gold docs):
                "answers a different question", so not useful by construction

Gold labels (proxies, no judge):
    useful      a real answer that doesn't abstain and has Answer Relevancy >= 0.8 in its pipeline run
    not useful  a real answer that opens by abstaining (its first sentence says "don't know", "couldn't
                find", ...), and every mismatched pair; a partial answer gets no gold label
    none        anything else (counts toward agreement only)

Reports agreement with the LLM, accuracy vs the gold labels for both, a TAU_USEFUL sweep for Jev from the
saved probabilities, disagreements, and latency/cost per call. Results go to evalres/useful_compare_<ts>.json.

Run from the project root:
    uv run python eval/useful_compare.py
    uv run python eval/useful_compare.py --real 60 --mismatched 20 --concurrency 5
"""
import argparse
import glob
import json
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_core.callbacks import UsageMetadataCallbackHandler

from decider_compare import ROOT, llm_price, pct  # noqa: E402  (also chdirs to the root and imports main)
from pipeline_eval import abstained  # noqa: E402

import decisions  # noqa: E402
import main  # noqa: E402

RESULTS_DIR = ROOT / "evalres"
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_pairs(n_real: int, n_mismatched: int, seed: int) -> list[dict]:
    questions = {json.loads(l)["qid"]: json.loads(l) for l in
                 (ROOT / "eval" / "eval_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    seen, real = set(), []
    for f in sorted(glob.glob(str(RESULTS_DIR / "pipeline_*.json"))):
        for r in json.loads(open(f, encoding="utf-8").read())["results"]:
            if "plan_queries" not in r["route"] or (r["qid"], r["answer"]) in seen:
                continue   # check_useful only runs on the knowledge-base route
            seen.add((r["qid"], r["answer"]))
            rel = ((r.get("metrics") or {}).get("Answer Relevancy") or {}).get("score")
            # "not useful" only when the answer opens by abstaining; a partial answer ("ID cards take 3 days;
            # I don't have information on what to bring") gets no gold label
            first = re.split(r"(?<=[.!?])\s|\n", r["answer"].strip(), maxsplit=1)[0]
            gold = (False if abstained(first) else None if abstained(r["answer"]) else
                    True if rel is not None and rel >= 0.8 else None)
            real.append({"kind": "real", "qid": r["qid"], "question": r["question"], "answer": r["answer"],
                         "gold": gold, "source": f.rsplit("/", 1)[-1]})
    rng = random.Random(seed)
    # every abstaining answer (the rare negatives) plus a random sample of the rest
    negatives = [p for p in real if p["gold"] is False]
    rest = [p for p in real if p["gold"] is not False]
    real = negatives + rng.sample(rest, max(0, min(len(rest), n_real - len(negatives))))
    mismatched = []
    answered = [p for p in real if p["gold"] is True]
    while len(mismatched) < n_mismatched and len(answered) > 1:
        a, b = rng.sample(answered, 2)
        if set(questions[a["qid"]]["gold_docs"]) & set(questions[b["qid"]]["gold_docs"]):
            continue   # same documents: the other answer might still help
        mismatched.append({"kind": "mismatched", "qid": f"{a['qid']}<-{b['qid']}", "question": a["question"],
                           "answer": b["answer"], "gold": False, "source": "mismatched"})
    return real + mismatched


def run_llm(p: dict, price) -> dict:
    usage = UsageMetadataCallbackHandler()
    started = time.perf_counter()
    try:
        grade = main.usefulness_checker.invoke({"question": p["question"], "answer": p["answer"]},
                                               config={"callbacks": [usage]})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    tokens = next(iter(usage.usage_metadata.values()), {})
    inp, out = tokens.get("input_tokens", 0), tokens.get("output_tokens", 0)
    return {"useful": grade.is_useful, "reason": grade.reason, "latency_s": round(time.perf_counter() - started, 3),
            "input_tokens": inp, "cost": inp * price["prompt"] + out * price["completion"] if price else None}


def run_jev(p: dict) -> dict:
    try:
        prob, meta = main.jev_useful(p["question"], p["answer"])
    except decisions.DecisionsUnavailable as e:
        return {"error": str(e)[:200]}
    return {"p": prob, **meta}


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r["llm"] and "error" not in r["jev"]]
    gold = [r for r in ok if r["gold"] is not None]

    def acc(rows, pick):
        return round(sum(pick(r) == r["gold"] for r in rows) / len(rows), 3) if rows else None

    def by_kind(pick):
        return {k: acc([r for r in gold if r["kind"] == k and r["gold"] is g], pick)
                for k, g in (("real", True), ("real", False), ("mismatched", False))} | {}

    def cl(key, cost_key="cost"):
        rows = [r[key] for r in ok]
        lat = [x["latency_s"] for x in rows]
        costs = [x[cost_key] for x in rows if x.get(cost_key) is not None]
        return {"latency_p50": pct(lat, 0.5), "latency_p95": pct(lat, 0.95),
                "cost_per_call": statistics.mean(costs) if costs else None}

    jev_at = lambda t: (lambda r: r["jev"]["p"] >= t)
    return {
        "n": len(results), "compared": len(ok), "with_gold": len(gold),
        "gold_counts": {"useful": sum(r["gold"] is True for r in gold), "not_useful": sum(r["gold"] is False for r in gold)},
        "agreement": round(sum(r["llm"]["useful"] == jev_at(main.TAU_USEFUL)(r) for r in ok) / len(ok), 3) if ok else None,
        "accuracy_vs_gold": {"llm": acc(gold, lambda r: r["llm"]["useful"]),
                             f"jev_at_{main.TAU_USEFUL}": acc(gold, jev_at(main.TAU_USEFUL))},
        "accuracy_by_group": {
            "llm": {"real useful": acc([r for r in gold if r["gold"] is True], lambda r: r["llm"]["useful"]),
                    "real not useful": acc([r for r in gold if r["kind"] == "real" and r["gold"] is False], lambda r: r["llm"]["useful"]),
                    "mismatched": acc([r for r in gold if r["kind"] == "mismatched"], lambda r: r["llm"]["useful"])},
            "jev": {"real useful": acc([r for r in gold if r["gold"] is True], jev_at(main.TAU_USEFUL)),
                    "real not useful": acc([r for r in gold if r["kind"] == "real" and r["gold"] is False], jev_at(main.TAU_USEFUL)),
                    "mismatched": acc([r for r in gold if r["kind"] == "mismatched"], jev_at(main.TAU_USEFUL))}},
        "tau_sweep": [{"tau": t, "accuracy": acc(gold, jev_at(t)),
                       "agreement_with_llm": round(sum(r["llm"]["useful"] == jev_at(t)(r) for r in ok) / len(ok), 3),
                       "false_not_useful": sum(r["gold"] is True and not jev_at(t)(r) for r in gold),
                       "false_useful": sum(r["gold"] is False and jev_at(t)(r) for r in gold)} for t in TAUS],
        "llm_errors": {"false_not_useful": sum(r["gold"] is True and not r["llm"]["useful"] for r in gold),
                       "false_useful": sum(r["gold"] is False and r["llm"]["useful"] for r in gold)},
        "cost_latency": {"llm": cl("llm"), "jev": cl("jev")},
    }


def print_report(s: dict, results: list[dict]):
    print(f"\nCompared {s['compared']}/{s['n']} pairs, {s['with_gold']} with a gold label {s['gold_counts']}")
    print(f"Agreement LLM vs Jev@{main.TAU_USEFUL}: {s['agreement']}")
    print("Accuracy vs gold:", s["accuracy_vs_gold"])
    print("By group:", json.dumps(s["accuracy_by_group"]))
    print("LLM errors:", s["llm_errors"])
    print(f"\n{'':6}{'p50 s':>8}{'p95 s':>8}{'$ / call':>12}")
    for k, c in s["cost_latency"].items():
        cost = f"{c['cost_per_call']:.7f}" if c["cost_per_call"] is not None else "n/a"
        print(f"{k:6}{c['latency_p50']:>8}{c['latency_p95']:>8}{cost:>12}")
    print(f"\n{'tau':>5}{'accuracy':>10}{'agree':>8}{'false not-useful':>18}{'false useful':>14}")
    for t in s["tau_sweep"]:
        print(f"{t['tau']:>5}{t['accuracy']:>10}{t['agreement_with_llm']:>8}{t['false_not_useful']:>18}{t['false_useful']:>14}")
    diffs = [r for r in results if "error" not in r["llm"] and "error" not in r["jev"]
             and r["llm"]["useful"] != (r["jev"]["p"] >= main.TAU_USEFUL)]
    print(f"\nDISAGREEMENTS ({len(diffs)}):")
    for r in diffs:
        print(f"\n  {r['qid']} [{r['kind']}] gold={r['gold']}  llm={r['llm']['useful']}  jev p={r['jev']['p']:.2f}")
        print(f"    Q: {r['question'][:120]}")
        print(f"    A: {r['answer'][:160]!r}")
        print(f"    LLM reason: {r['llm']['reason'][:140]}")


def main_eval(args):
    pairs = load_pairs(args.real, args.mismatched, args.seed)
    price = llm_price()
    print(f"Comparing usefulness checks on {len(pairs)} pairs: LLM={main.OPENROUTER['model']} (low effort) "
          f"vs Jev={decisions.DECISIONS_MODEL}")

    def run(p):
        r = {**p, "llm": run_llm(p, price), "jev": run_jev(p)}
        print(f"  {r['qid']:12} gold={str(r['gold']):5} llm={r['llm'].get('useful', 'ERR')!s:5} "
              f"jev p={r['jev'].get('p', float('nan')):.2f}")
        return r

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, pairs))
    summary = summarize(results)
    print_report(summary, results)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"useful_compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "stage": "useful_compare", "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"llm": main.OPENROUTER["model"], "llm_effort": "low", "llm_price_per_token": price,
                   "decisions_model": decisions.DECISIONS_MODEL, "tau_useful": main.TAU_USEFUL,
                   "real": args.real, "mismatched": args.mismatched, "seed": args.seed},
        "summary": summary, "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=int, default=120, help="saved real answers (all abstaining ones included)")
    parser.add_argument("--mismatched", type=int, default=40, help="question + another question's answer")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=5)
    main_eval(parser.parse_args())
