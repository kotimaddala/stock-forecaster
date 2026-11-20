"""Daily price and volume data.

Live sources, in order: yfinance (if installed, e.g. in Colab) then Stooq's
free CSV endpoint. Both are free and can be delayed or rate-limited, which is
why every series is checked for staleness before use.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Optional

import pandas as pd

from .http import CachedHttp, DataUnavailable

STOOQ_URL = "https://stooq.com/q/d/l/"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    needed = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataUnavailable(f"price data missing columns {missing}")
    df = df[needed].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


class MarketData:
    def __init__(self, http: CachedHttp) -> None:
        self.http = http

    def daily(self, ticker: str, start: Optional[str] = None) -> pd.DataFrame:
        df = self._from_yfinance(ticker, start)
        if df is None:
            df = self._from_stooq(ticker)
        if start:
            df = df[df["date"] >= date.fromisoformat(start)].reset_index(drop=True)
        if df.empty:
            raise DataUnavailable(f"no price data for {ticker}")
        return df

    def _from_yfinance(self, ticker: str, start: Optional[str]) -> Optional[pd.DataFrame]:
        if self.http.offline:
            return None
        try:
            import yfinance as yf  # optional dependency
        except ImportError:
            return None
        try:
            raw = yf.download(ticker, start=start or "2023-01-01", progress=False, auto_adjust=False)
        except Exception:
            return None
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index().rename(columns={"Date": "date"})
        return _normalize(raw)

    def _from_stooq(self, ticker: str) -> pd.DataFrame:
        symbol = ticker.lower().replace(".", "-")
        if not symbol.startswith("^") and "." not in symbol:
            symbol = f"{symbol}.us"
        text = self.http.get_text(STOOQ_URL, params={"s": symbol, "i": "d"})
        if text.strip().lower().startswith("no data") or "," not in text[:50]:
            raise DataUnavailable(f"Stooq returned no data for {ticker}")
        return _normalize(pd.read_csv(io.StringIO(text)))
