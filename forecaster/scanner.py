"""Daily short-term scan: a sequential pipeline (scan -> plan -> risk check -> route).

Signals: relative volume, a breakout above the prior 20-day high, and price
above the 50-day average. Every idea is a full plan with entry, stop, target
and size; an idea without a valid stop is never shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .config import RiskLimits
from .guardrails import check_trade_plan
from .indicators import add_indicators, size_position


@dataclass
class TradeIdea:
    ticker: str
    as_of: str
    entry: float
    stop: float
    target: float
    shares: int
    risk_pct: float
    position_pct: float
    reasons: List[str]
    route: str = "human"  # any new position needs approval
    violations: List[str] = field(default_factory=list)


def market_regime(spy: pd.DataFrame) -> Dict[str, object]:
    """A shock caps conviction and halves position sizes for the day."""
    df = add_indicators(spy)
    last, prev = df.iloc[-1], df.iloc[-2]
    day_move = last["close"] / prev["close"] - 1
    below_200 = pd.notna(last["sma200"]) and last["close"] < last["sma200"]
    shock = day_move <= -0.03 or below_200
    return {"shock": bool(shock), "day_move": round(float(day_move), 4), "below_200dma": bool(below_200)}


def scan(prices: Dict[str, pd.DataFrame], account: float, limits: RiskLimits,
         sector_exposure: Optional[Dict[str, float]] = None, sectors: Optional[Dict[str, str]] = None,
         market_shock: bool = False, min_rel_volume: float = 2.0) -> List[TradeIdea]:
    ideas: List[TradeIdea] = []
    sector_exposure = sector_exposure or {}
    sectors = sectors or {}
    for ticker, raw in prices.items():
        if len(raw) < 60:
            continue
        df = add_indicators(raw)
        row = df.iloc[-1]
        if row["close"] < limits.min_price or row["dollar_vol20"] < limits.min_avg_dollar_volume:
            continue  # liquidity and penny-stock floors
        reasons = []
        if row["rel_volume"] >= min_rel_volume:
            reasons.append(f"volume {row['rel_volume']:.1f}x its 20-day average")
        breakout = row["close"] > row["high20_prior"]
        if breakout:
            reasons.append(f"closed above the prior 20-day high ({row['high20_prior']:.2f})")
        if pd.notna(row["sma50"]) and row["close"] > row["sma50"]:
            reasons.append("above the 50-day average")
        if not (breakout and row["rel_volume"] >= min_rel_volume and len(reasons) == 3):
            continue
        entry = round(float(row["close"]), 2)
        stop = round(max(float(row["low10"]), entry - 1.5 * float(row["atr14"])), 2)
        if stop >= entry:
            continue
        target = round(entry + 2 * (entry - stop), 2)
        size = size_position(entry, stop, account, limits.max_risk_per_trade_pct, limits.max_position_pct,
                             scale=0.5 if market_shock else 1.0)
        if market_shock:
            reasons.append("market shock: size halved")
        sector = sectors.get(ticker, "unknown")
        report = check_trade_plan(entry, stop, size.shares, account, sector_exposure.get(sector, 0.0), limits)
        ideas.append(TradeIdea(ticker, str(row["date"]), entry, stop, target, size.shares, size.risk_pct,
                               size.position_pct, reasons, route="blocked" if not report.ok else "human",
                               violations=report.violations))
    ideas.sort(key=lambda i: i.risk_pct, reverse=True)
    return ideas
