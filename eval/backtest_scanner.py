"""Point-in-time backtest of the daily scanner (decision outcomes, not reasoning accuracy).

Each day, the scanner sees only data up to that day. A signal is entered at
the next day's open; it exits at the stop, the target, or after 20 trading
days. Results are compared with holding SPY over the same holding periods.

Usage (live data, run in Colab or locally with internet):
    python -m eval.backtest_scanner --start 2024-01-01 --tickers AAPL MSFT NVDA ...
Writes eval/results/backtest_results.md and backtest_trades.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from forecaster.config import RiskLimits, load_settings
from forecaster.scanner import scan

OUT = Path(__file__).parent / "results"
DEFAULT_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "AMD", "TSLA", "JPM", "V", "MA",
                    "UNH", "LLY", "COST", "HD", "XOM", "CAT", "GE", "ETN", "PWR", "VRT", "NEE", "ORCL", "CRM",
                    "ADBE", "NOW", "PANW", "UBER", "ANET"]


def simulate(prices: Dict[str, pd.DataFrame], benchmark: pd.DataFrame, start: str, account: float = 100_000,
             max_hold: int = 20, limits: RiskLimits = RiskLimits()) -> pd.DataFrame:
    trades: List[dict] = []
    dates = sorted(d for d in benchmark["date"] if str(d) >= start)
    bench = benchmark.set_index("date")
    open_until: Dict[str, object] = {}
    for i, day in enumerate(dates[:-1]):
        window = {t: df[df["date"] <= day] for t, df in prices.items()
                  if t not in open_until or open_until[t] < day}
        for idea in scan(window, account, limits):
            if idea.route == "blocked":
                continue
            future = prices[idea.ticker]
            future = future[future["date"] > day].head(max_hold)
            if future.empty:
                continue
            entry = float(future.iloc[0]["open"])
            if entry <= idea.stop:
                continue  # gapped through the stop before entry; skip
            exit_price, exit_day, reason = float(future.iloc[-1]["close"]), future.iloc[-1]["date"], "time"
            for _, bar in future.iterrows():
                if bar["low"] <= idea.stop:
                    exit_price, exit_day, reason = min(idea.stop, float(bar["open"])), bar["date"], "stop"
                    break
                if bar["high"] >= idea.target:
                    exit_price, exit_day, reason = max(idea.target, float(bar["open"])), bar["date"], "target"
                    break
            r_mult = (exit_price - entry) / (entry - idea.stop)
            b0 = float(bench.loc[future.iloc[0]["date"], "open"]) if future.iloc[0]["date"] in bench.index else None
            b1 = float(bench.loc[exit_day, "close"]) if exit_day in bench.index else None
            trades.append({"ticker": idea.ticker, "signal_day": day, "entry": round(entry, 2),
                           "stop": idea.stop, "target": idea.target, "exit": round(exit_price, 2),
                           "exit_day": exit_day, "reason": reason, "r_multiple": round(r_mult, 2),
                           "return_pct": round((exit_price / entry - 1) * 100, 2),
                           "dollar_pnl": round(idea.shares * (exit_price - entry), 2),
                           "spy_return_pct": round((b1 / b0 - 1) * 100, 2) if b0 and b1 else None})
            open_until[idea.ticker] = exit_day
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame, account: float = 100_000) -> str:
    if trades.empty:
        return "# Scanner backtest\n\nNo signals in the period.\n"
    pnl = trades.sort_values("exit_day")["dollar_pnl"].cumsum()
    equity = pd.concat([pd.Series([0.0]), pnl], ignore_index=True) + account
    drawdown = ((equity / equity.cummax()) - 1).min() * 100
    lines = ["# Scanner backtest (point-in-time)", "",
             f"Signals from {str(trades['signal_day'].min())[:10]} to {str(trades['exit_day'].max())[:10]}. "
             "Dollar amounts are written as USD so Markdown viewers do not read $ as math.", "",
             "| Measure | Result |", "| --- | --- |",
             f"| Trades | {len(trades)} |",
             f"| Win rate | {(trades['return_pct'] > 0).mean():.0%} |",
             f"| Average result per trade (in units of risk, R) | {trades['r_multiple'].mean():.2f} |",
             f"| Average trade return | {trades['return_pct'].mean():.2f}% |",
             f"| SPY return over the same holding periods | {trades['spy_return_pct'].mean():.2f}% |",
             f"| Exits: stop / target / time | {(trades['reason'] == 'stop').sum()} / "
             f"{(trades['reason'] == 'target').sum()} / {(trades['reason'] == 'time').sum()} |",
             f"| Total P&L on a {account:,.0f} USD paper account | {trades['dollar_pnl'].sum():,.0f} USD |",
             f"| Maximum drawdown of the trade equity curve | {drawdown:.1f}% |", "",
             "Costs, slippage and taxes are not modeled; results are directional, not proof."]
    return "\n".join(lines) + "\n"


def main() -> None:
    from forecaster.data import CachedHttp, MarketData
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    args = parser.parse_args()
    settings = load_settings()
    md = MarketData(CachedHttp(settings.cache_dir))
    warmup = str(pd.Timestamp(args.start) - pd.Timedelta(days=320))[:10]
    prices = {}
    for t in args.tickers:
        try:
            prices[t] = md.daily(t, start=warmup)
        except Exception as exc:
            print(f"skip {t}: {exc}")
    spy = md.daily("SPY", start=warmup)
    trades = simulate(prices, spy, args.start)
    OUT.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT / "backtest_trades.csv", index=False)
    (OUT / "backtest_results.md").write_text(report(trades))
    print(report(trades))


if __name__ == "__main__":
    main()
