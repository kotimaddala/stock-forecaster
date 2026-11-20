"""Live evaluation: real models through OpenRouter on the same fixed scenarios.

Compares the Tree-of-Thoughts multi-agent review with a single chain-of-thought
baseline. Both see identical facts and documents, so differences come from the
reasoning design, not the data.

Usage (needs OPENROUTER_API_KEY):
    python -m eval.run_live_eval --repeats 3
Writes eval/results/live_results.md and live_results.json.
Cost: roughly 5-10 model calls per ToT review; set cheap models in .env to test first.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from forecaster.config import load_settings
from forecaster.llm import OpenRouterLLM

from .harness import isolation_violations, run
from .scenario_data import all_scenarios

OUT = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="scenario ids, e.g. S01 S07")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.openrouter_api_key:
        raise SystemExit("Set OPENROUTER_API_KEY (see .env.example)")
    scenarios = [s for s in all_scenarios() if s["live_eligible"] and (not args.only or s["id"] in args.only)]
    rows = []
    for scn in scenarios:
        for mode in ("tot", "single"):
            for rep in range(args.repeats):
                llm = OpenRouterLLM(api_key=settings.openrouter_api_key)
                try:
                    result = run(scn, mode=mode, llm=llm, settings=settings)
                except Exception as exc:  # record failures instead of stopping the run
                    rows.append({"id": scn["id"], "mode": mode, "rep": rep, "error": str(exc)})
                    continue
                rec = result.recommendation
                selected = [b for b in result.branches if b.status == "selected"]
                rows.append({
                    "id": scn["id"], "mode": mode, "rep": rep, "action": rec.action,
                    "correct": rec.action in scn["acceptable_actions"], "route": rec.route,
                    "fallback": rec.fallback, "violations": result.guardrails.violations,
                    "leaks": isolation_violations(scn, result), "llm_calls": result.llm_calls,
                    "seconds": result.seconds,
                    "reviewer_confidence": selected[0].reviewer_confidence if selected else None,
                    "explanation": rec.explanation,
                })
                print(scn["id"], mode, rep, rec.action, rec.route, f"{result.seconds}s")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (OUT / "live_results.json").write_text(json.dumps({"settings": {
        "analyst_model": settings.analyst_model, "reviewer_model": settings.reviewer_model},
        "summary": summary, "rows": rows}, indent=2))
    (OUT / "live_results.md").write_text(render(summary, settings))
    print(json.dumps(summary, indent=2))


def summarize(rows):
    out = {}
    for mode in ("tot", "single"):
        ok = [r for r in rows if r["mode"] == mode and "error" not in r]
        if not ok:
            continue
        out[mode] = {
            "runs": len(ok),
            "errors": sum(1 for r in rows if r["mode"] == mode and "error" in r),
            "acceptable_action_rate": round(sum(r["correct"] for r in ok) / len(ok), 3),
            # A blocked recommendation never reaches the user, so it cannot count as a usable answer.
            "acceptable_and_delivered_rate": round(sum(r["correct"] and r["route"] != "blocked" for r in ok) / len(ok), 3),
            "blocked_or_violations": sum(1 for r in ok if r["violations"]),
            "fallbacks": sum(r["fallback"] for r in ok),
            "evidence_leaks": sum(len(r["leaks"]) for r in ok),
            "median_llm_calls": statistics.median(r["llm_calls"] for r in ok),
            "median_seconds": round(statistics.median(r["seconds"] for r in ok), 1),
            "escalation_rate": round(sum(r["route"] == "human" for r in ok) / len(ok), 3),
        }
        by_scn = {}
        for r in ok:
            by_scn.setdefault(r["id"], set()).add(r["action"])
        repeated = [sid for sid in by_scn if sum(1 for r in ok if r["id"] == sid) > 1]
        if repeated:
            same = sum(1 for sid in repeated if len(by_scn[sid]) == 1)
            out[mode]["same_action_across_repeats"] = f"{same}/{len(repeated)} scenarios"
        conf = [(r["reviewer_confidence"], r["correct"]) for r in ok if r.get("reviewer_confidence") is not None]
        if conf:
            right = [c for c, k in conf if k]
            wrong = [c for c, k in conf if not k]
            out[mode]["mean_confidence_when_right"] = round(statistics.mean(right), 2) if right else None
            out[mode]["mean_confidence_when_wrong"] = round(statistics.mean(wrong), 2) if wrong else None
    per = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r["id"], r["mode"])
        per.setdefault(key, []).append((r["correct"], r["action"], r["route"]))
    out["per_scenario"] = {f"{k[0]}:{k[1]}": f"{sum(c for c, _, _ in v)}/{len(v)} ("
                           + ", ".join(f"{a}/{rt}" for _, a, rt in v) + ")" for k, v in sorted(per.items())}
    return out


def render(summary, settings) -> str:
    lines = ["# Live evaluation results", "",
             f"Analyst model: `{settings.analyst_model}`; Reviewer model: `{settings.reviewer_model}`.", "",
             "| Measure | Tree of Thoughts (multi-agent) | Single chain of thought |", "| --- | --- | --- |"]
    keys = ["runs", "acceptable_action_rate", "acceptable_and_delivered_rate", "same_action_across_repeats",
            "escalation_rate", "blocked_or_violations", "fallbacks",
            "evidence_leaks", "median_llm_calls", "median_seconds", "mean_confidence_when_right",
            "mean_confidence_when_wrong", "errors"]
    for k in keys:
        lines.append(f"| {k.replace('_', ' ')} | {summary.get('tot', {}).get(k, '-')} | "
                     f"{summary.get('single', {}).get(k, '-')} |")
    lines += ["", "## Per scenario (acceptable actions / runs, then action/route of each run)", ""]
    lines += [f"- {k}: {v}" for k, v in summary.get("per_scenario", {}).items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
