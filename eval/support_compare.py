"""Support-check gate before/after: the LLM support_checker vs a Decisions-model (Jev) gate, same inputs.

With SUPPORT=jev, check_support asks Jev first: if p(fully supported) >= TAU_SUP the answer passes without the
LLM; otherwise the LLM check runs as before. So the gate can only make two kinds of difference:
    saved LLM call    Jev passes an answer the LLM would also pass (the goal)
    false pass        Jev passes an answer the LLM would have caught (the risk: faithfulness)

Inputs are (question, chunks, answer) triples; nothing is generated:
    real          distinct answers saved in evalres/pipeline_*.json with the chunks the generator saw
                  (knowledge-base route; flagged and contact answers left out: they carry appended text)
    number        a real answer with one number that also appears in its chunks changed (12 -> 15)
    wrong_context a real answer checked against another question's chunks (no shared doc_ids)
    workaround    a real answer with a made-up exception appended (the case the LLM check was tuned for)
The three synthetic kinds are unsupported by construction. Real answers have no gold label: the LLM's verdict
is the reference.

Reports, per TAU_SUP (replayed from the saved probabilities): share of real answers that skip the LLM, false
passes on real answers (Jev passes, LLM says not fully) and on synthetic ones, plus latency and cost.
Results go to evalres/support_compare_<timestamp>.json.

Run from the project root:
    uv run python eval/support_compare.py
    uv run python eval/support_compare.py --real 40 --synthetic 10 --concurrency 5
"""
import argparse
import glob
import json
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document

from decider_compare import ROOT, llm_price, pct  # noqa: E402  (also chdirs to the root and imports main)

import decisions  # noqa: E402
import main  # noqa: E402

RESULTS_DIR = ROOT / "evalres"
TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
WORKAROUND = ("\n- Exception: your skip-level manager can waive this rule if you email them a short "
              "justification, and the waiver applies immediately.")
NUMBER = re.compile(r"(?<![\w.])\d+(?![\w])")


def load_real() -> list[dict]:
    seen, rows = set(), []
    for f in sorted(glob.glob(str(RESULTS_DIR / "pipeline_*.json"))):
        for r in json.loads(open(f, encoding="utf-8").read())["results"]:
            if ("plan_queries" not in r["route"] or not r.get("context") or "flagged" in r["outcome"]
                    or "contact" in r["outcome"] or (r["qid"], r["answer"]) in seen):
                continue
            seen.add((r["qid"], r["answer"]))
            rows.append({"kind": "real", "qid": r["qid"], "question": r["question"], "answer": r["answer"],
                         "chunks": r["context"], "gold": None})
    return rows


def mutate_number(p: dict, rng) -> dict | None:
    context = " ".join(c["text"] for c in p["chunks"])
    # numbers inside a doc_id citation ([HR-LV-001]) don't count: citations aren't claims
    cited = [c.span() for c in re.finditer(r"[\[【][A-Z][A-Z0-9-]*[\]】]", p["answer"])]
    candidates = [m for m in NUMBER.finditer(p["answer"]) if re.search(rf"(?<!\d){m.group()}(?!\d)", context)
                  and m.group() not in p["question"] and not any(a <= m.start() < b for a, b in cited)]
    if not candidates:
        return None
    m = rng.choice(candidates)
    n = int(m.group())
    new = next((str(v) for v in (n + 3, n + 7, n * 2 + 1, n + 11)
                if not re.search(rf"(?<!\d){v}(?!\d)", context)), None)
    if new is None:   # every replacement also appears in the chunks: it might still be supported
        return None
    answer = p["answer"][:m.start()] + new + p["answer"][m.end():]
    return {**p, "kind": "number", "answer": answer, "gold": False, "change": f"{m.group()} -> {new}"}


def make_synthetic(real: list[dict], n: int, rng) -> list[dict]:
    pool = real[:]
    rng.shuffle(pool)
    number = [x for x in (mutate_number(p, rng) for p in pool) if x][:n]
    wrong = []
    for p in pool:
        if len(wrong) >= n:
            break
        ids = {c["doc_id"] for c in p["chunks"]}
        other = next((o for o in pool if o["qid"] != p["qid"] and not ids & {c["doc_id"] for c in o["chunks"]}), None)
        if other:
            wrong.append({**p, "kind": "wrong_context", "chunks": other["chunks"], "gold": False,
                          "change": f"chunks of {other['qid']}"})
    workaround = [{**p, "kind": "workaround", "answer": p["answer"].rstrip() + WORKAROUND, "gold": False,
                   "change": "appended made-up exception"} for p in pool
                  if "skip-level" not in " ".join(c["text"] for c in p["chunks"]).lower()][:n]
    return number + wrong + workaround


def docs_of(p: dict) -> list[Document]:
    return [Document(page_content=c["text"], metadata={k: c.get(k) for k in ("doc_id", "title", "status")})
            for c in p["chunks"]]


def run_llm(p: dict, price) -> dict:
    usage = UsageMetadataCallbackHandler()
    started = time.perf_counter()
    try:
        grade = main.support_checker.invoke({"context": main.format_docs(docs_of(p)), "question": p["question"],
                                             "answer": p["answer"]}, config={"callbacks": [usage]})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    tokens = next(iter(usage.usage_metadata.values()), {})
    inp, out = tokens.get("input_tokens", 0), tokens.get("output_tokens", 0)
    support = grade.support if grade.unsupported_claims else "fully"   # as check_support does
    return {"fully": support == "fully", "support": support, "claims": grade.unsupported_claims,
            "latency_s": round(time.perf_counter() - started, 3), "input_tokens": inp, "output_tokens": out,
            "cost": inp * price["prompt"] + out * price["completion"] if price else None}


def run_jev(p: dict) -> dict:
    try:
        prob, meta = main.jev_support(p["question"], docs_of(p), p["answer"])
    except decisions.DecisionsUnavailable as e:
        return {"error": str(e)[:200]}
    return {"p": prob, **meta}


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r["llm"] and "error" not in r["jev"]]
    real = [r for r in ok if r["kind"] == "real"]
    synth = [r for r in ok if r["kind"] != "real"]
    kinds = sorted({r["kind"] for r in synth})

    def at(t):
        passed = lambda r: r["jev"]["p"] >= t
        return {"tau": t,
                "real_skip_llm": round(sum(passed(r) for r in real) / len(real), 3) if real else None,
                "real_false_pass": sum(passed(r) and not r["llm"]["fully"] for r in real),
                "synthetic_false_pass": {k: sum(passed(r) for r in synth if r["kind"] == k) for k in kinds},
                # with the gate, an unsupported answer is caught only if Jev doesn't pass it AND the LLM catches it
                "synthetic_caught_with_gate": {k: sum(not passed(r) and not r["llm"]["fully"] for r in synth
                                                      if r["kind"] == k) for k in kinds}}

    def cl(key):
        rows = [r[key] for r in ok]
        lat = [x["latency_s"] for x in rows]
        costs = [x["cost"] for x in rows if x.get("cost") is not None]
        return {"latency_p50": pct(lat, 0.5), "latency_p95": pct(lat, 0.95),
                "latency_mean": round(statistics.mean(lat), 3) if lat else None,
                "cost_per_call": statistics.mean(costs) if costs else None}

    llm, jev = cl("llm"), cl("jev")
    skip = at(main.TAU_SUP)["real_skip_llm"] or 0
    return {
        "n": len(results), "compared": len(ok), "real": len(real),
        "synthetic": {k: sum(r["kind"] == k for r in synth) for k in kinds},
        "llm_on_real": {"fully": sum(r["llm"]["fully"] for r in real), "not_fully": sum(not r["llm"]["fully"] for r in real)},
        "llm_caught_synthetic": {k: sum(not r["llm"]["fully"] for r in synth if r["kind"] == k) for k in kinds},
        "at_tau": at(main.TAU_SUP), "tau_sweep": [at(t) for t in TAUS],
        "cost_latency": {"llm": llm, "jev": jev,
                         "gated_mean_latency_on_real": round(jev["latency_mean"] + (1 - skip) * llm["latency_mean"], 3)
                         if real else None},
    }


def print_report(s: dict, results: list[dict]):
    print(f"\nCompared {s['compared']}/{s['n']}: {s['real']} real answers, synthetic {s['synthetic']}")
    print(f"LLM on real answers: {s['llm_on_real']}")
    print(f"LLM catches synthetic (alone): {s['llm_caught_synthetic']}")
    c = s["cost_latency"]
    print(f"\n{'':6}{'p50 s':>8}{'p95 s':>8}{'mean s':>8}{'$ / call':>12}")
    for k in ("llm", "jev"):
        cost = f"{c[k]['cost_per_call']:.7f}" if c[k]["cost_per_call"] is not None else "n/a"
        print(f"{k:6}{c[k]['latency_p50']:>8}{c[k]['latency_p95']:>8}{c[k]['latency_mean']:>8}{cost:>12}")
    print(f"Gated mean latency on real answers at TAU_SUP={main.TAU_SUP}: {c['gated_mean_latency_on_real']} s "
          f"(LLM alone {c['llm']['latency_mean']} s)")
    print(f"\n{'tau':>5}{'skip LLM':>10}{'real false pass':>17}   synthetic false pass / caught with gate")
    for t in s["tau_sweep"]:
        print(f"{t['tau']:>5}{t['real_skip_llm']:>10}{t['real_false_pass']:>17}   "
              f"{t['synthetic_false_pass']} / {t['synthetic_caught_with_gate']}")
    flagged = [r for r in results if "error" not in r["jev"] and "error" not in r["llm"] and r["jev"]["p"] >= main.TAU_SUP
               and (r["kind"] != "real" or not r["llm"]["fully"])]
    print(f"\nFALSE PASSES at TAU_SUP={main.TAU_SUP} ({len(flagged)}):")
    for r in flagged:
        print(f"\n  {r['qid']} [{r['kind']}] {r.get('change', '')}  jev p={r['jev']['p']:.2f}  llm={r['llm']['support']}")
        print(f"    Q: {r['question'][:120]}")
        print(f"    A: {r['answer'][:200]!r}")
        if r["llm"]["claims"]:
            print(f"    LLM unsupported: {r['llm']['claims'][:2]}")


def main_eval(args):
    rng = random.Random(args.seed)
    real = load_real()
    real = rng.sample(real, min(args.real, len(real)))
    pairs = real + make_synthetic(real, args.synthetic, rng)
    price = llm_price()
    print(f"Comparing support checks on {len(pairs)} triples: LLM={main.OPENROUTER['model']} (medium effort) "
          f"vs Jev gate={decisions.DECISIONS_MODEL}")

    def run(p):
        r = {**p, "llm": run_llm(p, price), "jev": run_jev(p)}
        print(f"  {r['qid']:6} {r['kind']:14} llm={r['llm'].get('support', 'ERR'):10} "
              f"jev p={r['jev'].get('p', float('nan')):.2f}")
        return r

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, pairs))
    summary = summarize(results)
    print_report(summary, results)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"support_compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "stage": "support_compare", "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"llm": main.OPENROUTER["model"], "llm_effort": "medium", "llm_price_per_token": price,
                   "decisions_model": decisions.DECISIONS_MODEL, "tau_sup": main.TAU_SUP,
                   "real": args.real, "synthetic_per_kind": args.synthetic, "seed": args.seed},
        "summary": summary, "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=int, default=80, help="saved real answers to check")
    parser.add_argument("--synthetic", type=int, default=25, help="unsupported answers per synthetic kind")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=5)
    main_eval(parser.parse_args())
