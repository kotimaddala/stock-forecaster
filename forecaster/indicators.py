"""Technical indicators and position sizing. All numbers come from code, never the model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma20"] = out["close"].rolling(20).mean()
    out["sma50"] = out["close"].rolling(50).mean()
    out["sma200"] = out["close"].rolling(200).mean()
    out["vol20"] = out["volume"].rolling(20).mean()
    out["rel_volume"] = out["volume"] / out["vol20"].shift(1)
    out["high20_prior"] = out["high"].rolling(20).max().shift(1)
    out["low10"] = out["low"].rolling(10).min()
    prev_close = out["close"].shift(1)
    true_range = pd.concat([out["high"] - out["low"], (out["high"] - prev_close).abs(),
                            (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = true_range.rolling(14).mean()
    out["dollar_vol20"] = (out["close"] * out["volume"]).rolling(20).mean()
    return out


def technical_rating(row: pd.Series, lagging_sector: bool = False) -> str:
    """Checkpoint 1.1 rules: weekly-close detail simplified to daily closes."""
    if pd.notna(row.get("sma200")) and row["close"] < row["sma200"]:
        return "broken"
    if (pd.notna(row.get("sma50")) and row["close"] < row["sma50"]) or lagging_sector:
        return "weakening"
    return "supportive"


@dataclass
class PositionSize:
    shares: int
    risk_dollars: float
    risk_pct: float
    position_pct: float
    capped_by: str = ""


def size_position(entry: float, stop: float, account: float, max_risk_pct: float,
                  max_position_pct: float, scale: float = 1.0) -> PositionSize:
    """Shares such that hitting the stop loses at most max_risk_pct of the account."""
    if stop <= 0 or stop >= entry:
        raise ValueError("stop must be positive and below entry")
    per_share = entry - stop
    by_risk = math.floor(account * max_risk_pct / 100 * scale / per_share)
    by_size = math.floor(account * max_position_pct / 100 * scale / entry)  # a shock shrinks both limits
    shares = max(0, min(by_risk, by_size))
    capped = "position cap" if by_size < by_risk else ""
    risk = shares * per_share
    return PositionSize(shares, round(risk, 2), round(risk / account * 100, 3),
                        round(shares * entry / account * 100, 2), capped)
