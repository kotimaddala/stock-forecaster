"""Command-line interface.

    python -m forecaster.cli demo                      # offline walkthrough, no keys needed
    python -m forecaster.cli review examples/thesis_example.json --trigger "10-Q filed"
    python -m forecaster.cli review ... --mode single  # chain-of-thought baseline
    python -m forecaster.cli scan --tickers AAPL MSFT NVDA
    python -m forecaster.cli decide 1 approve agree    # record a human decision
    python -m forecaster.cli monitor                   # journal summary for the dashboard
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings
from .journal import Journal
from .review_card import render_card


def cmd_demo(args, settings) -> int:
    from eval.harness import run
    from eval.scenario_data import all_scenarios
    scn = all_scenarios()[0]
    result = run(scn)
    print(f"Scenario: {scn['title']}\n")
    for b in result.branches:
        print(f"  [{b.status:>16}] {b.branch_id:<3} score {b.score:>4}  {b.explanation}"
              + (f"  ({b.prune_reason})" if b.prune_reason else ""))
    print("\n" + render_card(result))
    return 0


def cmd_review(args, settings) -> int:
    from .pipeline import run_live_review
    journal = Journal(settings.journal_path)
    result, notes = run_live_review(args.thesis, args.trigger, settings, mode=args.mode, journal=journal,
                                    market_shock=args.shock, offline=args.offline)
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    print(render_card(result))
    if args.json:
        print(json.dumps(result.recommendation.to_dict(), indent=2))
    print(f"\n{result.llm_calls} model calls, {result.retrieval_calls} retrieval calls, {result.seconds}s",
          file=sys.stderr)
    return 0


def cmd_scan(args, settings) -> int:
    from .data import CachedHttp, MarketData
    from .scanner import market_regime, scan
    md = MarketData(CachedHttp(settings.cache_dir))
    prices = {}
    for t in args.tickers:
        try:
            prices[t] = md.daily(t, start="2025-01-01")
        except Exception as exc:
            print(f"skip {t}: {exc}", file=sys.stderr)
    regime = market_regime(md.daily("SPY", start="2025-01-01"))
    print(f"Market regime: {regime}")
    ideas = scan(prices, args.account, settings.risk, market_shock=regime["shock"])
    if not ideas:
        print("No ideas passed the scan today.")
    for i in ideas:
        print(f"{i.ticker}: entry {i.entry} stop {i.stop} target {i.target} shares {i.shares} "
              f"(risk {i.risk_pct}%, size {i.position_pct}%) route={i.route} | {'; '.join(i.reasons)}"
              + (f" | BLOCKED: {i.violations}" if i.violations else ""))
    return 0


def cmd_decide(args, settings) -> int:
    Journal(settings.journal_path).record_decision(args.review_id, args.decision, args.reason, args.note or "")
    print("recorded")
    return 0


def cmd_monitor(args, settings) -> int:
    print(json.dumps(Journal(settings.journal_path).summary(), indent=2))
    return 0


def main(argv=None) -> int:
    settings = load_settings()
    p = argparse.ArgumentParser(prog="forecaster", description="Stock Opportunity Forecaster (paper trading only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="offline walkthrough of the S01 scenario")
    r = sub.add_parser("review", help="review a thesis with live public data")
    r.add_argument("thesis")
    r.add_argument("--trigger", default="Scheduled review")
    r.add_argument("--mode", choices=["tot", "single"], default="tot")
    r.add_argument("--shock", action="store_true", help="treat the market as in a shock")
    r.add_argument("--offline", action="store_true", help="use only cached data")
    r.add_argument("--json", action="store_true")
    s = sub.add_parser("scan", help="daily breakout scan")
    s.add_argument("--tickers", nargs="+", required=True)
    s.add_argument("--account", type=float, default=100_000)
    d = sub.add_parser("decide", help="record your decision on a routed review")
    d.add_argument("review_id", type=int)
    d.add_argument("decision", choices=["approve", "approve_with_changes", "skip"])
    d.add_argument("reason")
    d.add_argument("--note")
    sub.add_parser("monitor", help="journal summary: routes, escalation rate, approvals, lessons")
    args = p.parse_args(argv)
    return {"demo": cmd_demo, "review": cmd_review, "scan": cmd_scan, "decide": cmd_decide,
            "monitor": cmd_monitor}[args.cmd](args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
