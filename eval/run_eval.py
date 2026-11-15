"""Offline evaluation: runs every scenario with scripted model responses.

What this measures: the control layer (search, pruning, decision rules,
guardrails, fallback, routing) and retrieval, on fixed synthetic inputs.
What it does NOT measure: how good real models are at proposing and judging
explanations. That is what eval/run_live_eval.py measures with OpenRouter.

Usage:  python -m eval.run_eval
Writes: eval/results/offline_results.md, offline_results.json, demo_S01.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from forecaster.models import Chunk
from forecaster.retrieval import VectorIndex
from forecaster.review_card import render_card

from .harness import check, isolation_violations, run
from .scenario_data import BASE_CHUNKS, RETRIEVAL_QUESTIONS, T, all_scenarios

OUT = Path(__file__).parent / "results"
SCN_DIR = Path(__file__).parent / "scenarios"


def retrieval_benchmark() -> dict:
    index = VectorIndex(min_score=0.08, top_k=5)
    index.add([Chunk(**c) for c in BASE_CHUNKS])
    rows, totals = [], Counter()
    for question, gold in RETRIEVAL_QUESTIONS:
        golds = {gold} if isinstance(gold, str) else set(gold)
        filtered = [e.chunk_id for e in index.search(question, ticker=T)]
        unfiltered = [e.chunk_id for e in index.search(question, ticker="*", include_superseded=True)]
        row = {
            "question": question, "gold": sorted(golds),
            "filtered_top1": filtered[:1], "unfiltered_top1": unfiltered[:1],
            "hit1_filtered": bool(set(filtered[:1]) & golds), "hit5_filtered": bool(set(filtered[:5]) & golds),
            "hit1_unfiltered": bool(set(unfiltered[:1]) & golds),
            "hit5_unfiltered": bool(set(unfiltered[:5]) & golds),
            "wrong_source_top1_unfiltered": bool(unfiltered[:1] and unfiltered[0] in ("V-MDA-1", "G-MDA-OLD")),
            "wrong_source_any_filtered": any(c in ("V-MDA-1", "G-MDA-OLD") for c in filtered),
        }
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, bool):
                totals[k] += int(v)
    n = len(rows)
    return {"n": n, "rows": rows, "rates": {k: round(v / n, 3) for k, v in totals.items()}}


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    SCN_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = all_scenarios()
    rows, by_cat = [], defaultdict(lambda: [0, 0])
    totals = Counter()
    for scn in scenarios:
        (SCN_DIR / f"{scn['id']}.json").write_text(json.dumps(scn, indent=2))
        result = run(scn)
        failures = check(scn, result)
        leaks = isolation_violations(scn, result)
        # Robustness: the same inputs in a different order must give the same decision.
        variants = [run(scn, shuffle_seed=seed).recommendation for seed in (1, 2, 3)]
        stable = all(v.action == result.recommendation.action and v.route == result.recommendation.route
                     for v in variants)
        rec = result.recommendation
        rows.append({
            "id": scn["id"], "title": scn["title"], "category": scn["category"], "pass": not failures,
            "failures": failures, "action": rec.action, "route": rec.route, "contested": rec.contested,
            "fallback": rec.fallback, "conviction": f"{rec.conviction_before}->{rec.conviction_after}",
            "llm_calls": result.llm_calls, "retrieval_calls": result.retrieval_calls,
            "violations": result.guardrails.violations, "isolation_leaks": leaks, "order_stable": stable,
            "route_reasons": rec.route_reasons,
        })
        by_cat[scn["category"]][0] += int(not failures)
        by_cat[scn["category"]][1] += 1
        totals["routes_" + rec.route] += 1

    n = len(rows)
    tot_runs = [r for r in rows if r["action"] and not r["fallback"]]
    summary = {
        "scenarios": n,
        "passed": sum(r["pass"] for r in rows),
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in by_cat.items()},
        "order_stable": sum(r["order_stable"] for r in rows),
        "isolation_leaks": sum(len(r["isolation_leaks"]) for r in rows),
        "escalation_rate": round(totals["routes_human"] / n, 3),
        "routes": {k[7:]: v for k, v in totals.items() if k.startswith("routes_")},
        "avg_llm_calls_full_reviews": round(sum(r["llm_calls"] for r in tot_runs) / max(1, len(tot_runs)), 1),
        "max_llm_calls": max(r["llm_calls"] for r in rows),
        "max_retrieval_calls": max(r["retrieval_calls"] for r in rows),
    }
    retrieval = retrieval_benchmark()
    payload = {"summary": summary, "scenarios": rows, "retrieval": retrieval}
    (OUT / "offline_results.json").write_text(json.dumps(payload, indent=2))
    (OUT / "offline_results.md").write_text(render_markdown(payload))
    write_demo(scenarios[0])
    print(json.dumps(summary, indent=2))
    print("retrieval:", json.dumps(retrieval["rates"], indent=2))
    return payload


def render_markdown(p: dict) -> str:
    s, r = p["summary"], p["retrieval"]["rates"]
    lines = [
        "# Offline evaluation results",
        "",
        "Generated by `python -m eval.run_eval`. Inputs are synthetic and model responses are scripted, so "
        "these results test the control layer and retrieval, not model judgment. See `run_live_eval.py` for "
        "the live comparison.",
        "",
        "## Summary",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Scenarios passing all expectations | {s['passed']}/{s['scenarios']} |",
        f"| By category | {', '.join(f'{k} {v}' for k, v in s['by_category'].items())} |",
        f"| Same decision when input order is shuffled (3 shuffles each) | {s['order_stable']}/{s['scenarios']} |",
        f"| Evidence from another company or a superseded filing used | {s['isolation_leaks']} |",
        f"| Routed to human review | {s['escalation_rate']:.0%} ({s['routes']}) |",
        f"| Model calls per full review (average / max) | {s['avg_llm_calls_full_reviews']} / {s['max_llm_calls']} (budget 14) |",
        f"| Retrieval calls per review (max) | {s['max_retrieval_calls']} (budget 12) |",
        "",
        "## Retrieval benchmark (10 questions, 8 passages including 2 distractors)",
        "",
        "| Setting | Correct passage ranked first | Correct passage in top 5 | Wrong company or outdated passage ranked first |",
        "| --- | --- | --- | --- |",
        f"| With filters (design) | {r['hit1_filtered']:.0%} | {r['hit5_filtered']:.0%} | 0% (never returned: {r['wrong_source_any_filtered']:.0%}) |",
        f"| Without filters (ablation) | {r['hit1_unfiltered']:.0%} | {r['hit5_unfiltered']:.0%} | {r['wrong_source_top1_unfiltered']:.0%} |",
        "",
        "## Scenarios",
        "",
        "| ID | Scenario | Category | Pass | Action | Route | Model calls | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in p["scenarios"]:
        note = "; ".join(row["failures"] or row["violations"] or row["route_reasons"][:2])
        lines.append(f"| {row['id']} | {row['title']} | {row['category']} | {'yes' if row['pass'] else 'NO'} | "
                     f"{row['action']} | {row['route']} | {row['llm_calls']} | {note} |")
    return "\n".join(lines) + "\n"


def write_demo(scn: dict) -> None:
    tot = run(scn)
    single = run(scn, mode="single")
    lines = ["# Demo: scenario S01 (synthetic)", "", scn["note"], "", "## Reasoning tree", "",
             "| Node | Depth | Explanation | Action | Score | Status | Reason |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for b in tot.branches:
        lines.append(f"| {b.branch_id} | {b.depth} | {b.explanation} | {b.implied_action} | {b.score} | "
                     f"{b.status} | {b.prune_reason} |")
    lines += ["", "## Controller trace", ""] + [f"- {t}" for t in tot.trace]
    lines += ["", "## Reviewer card (what the human sees)", "", render_card(tot)]
    lines += ["", "## Single chain-of-thought baseline on the same inputs (illustration, scripted)", "",
              f"- Action: **{single.recommendation.action}**, conviction {single.recommendation.conviction_before}"
              f" -> {single.recommendation.conviction_after}",
              f"- Explanation: {single.recommendation.explanation}",
              "- It accepted the press release's story and never looked for a second cause. The live evaluation "
              "measures whether real models show this pattern."]
    (OUT / "demo_S01.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
