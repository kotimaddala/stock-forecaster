"""MCP server: exposes the forecaster as tools for Claude Code or Gemini CLI (Module 7 plugin lab).

Run:  python -m forecaster.mcp_server   (requires `pip install mcp`)

Deliberately read-only: there is no order-placement tool. Agents in other
hosts can research and review, but cannot trade; approval stays with the
owner through the journal and broker gate.
"""

from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)  # stdout carries the protocol


def build_app():
    from mcp.server.fastmcp import FastMCP  # optional dependency

    from .config import load_settings
    from .journal import Journal
    from .review_card import render_card

    app = FastMCP(name="stock-forecaster")
    settings = load_settings()

    @app.tool()
    def review_thesis(thesis_path: str, trigger: str = "Scheduled review", mode: str = "tot") -> str:
        """Review a saved investment thesis against the latest SEC filings, prices and policy documents.
        Returns the reviewer card: recommendation, route, evidence links and what would change the view.
        Paper trading only; nothing is executed."""
        from .pipeline import run_live_review
        result, notes = run_live_review(thesis_path, trigger, settings, mode=mode,
                                        journal=Journal(settings.journal_path))
        return render_card(result) + ("\n\nData notes: " + "; ".join(notes) if notes else "")

    @app.tool()
    def scan_breakouts(tickers: list[str], account: float = 100_000) -> str:
        """Run the daily breakout scan. Every idea includes entry, stop-loss, target and size, and
        requires human approval."""
        from .data import CachedHttp, MarketData
        from .scanner import scan
        md = MarketData(CachedHttp(settings.cache_dir))
        prices = {t: md.daily(t, start="2025-01-01") for t in tickers}
        ideas = scan(prices, account, settings.risk)
        return json.dumps([i.__dict__ for i in ideas], indent=2, default=str) or "[]"

    @app.tool()
    def journal_summary() -> str:
        """Monitoring summary: review routes, escalation rate, approval rate, lesson stages."""
        return json.dumps(Journal(settings.journal_path).summary(), indent=2)

    @app.tool()
    def offline_demo() -> str:
        """Run the synthetic S01 walkthrough (no keys needed) and return the reviewer card."""
        from eval.harness import run
        from eval.scenario_data import all_scenarios
        return render_card(run(all_scenarios()[0]))

    return app


def main() -> None:
    try:
        build_app().run()
    except (BrokenPipeError, EOFError):
        sys.exit(0)


if __name__ == "__main__":
    main()
