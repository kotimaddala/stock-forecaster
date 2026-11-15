"""Live pipeline: build facts and the retrieval index from public sources, then review.

Sources (all free): SEC EDGAR (figures and filing text), Federal Register
(policy status and text), Stooq or yfinance (prices). Results are cached so a
run can be repeated offline.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import Settings
from .controller import ReviewResult, ThesisReviewer
from .data import CachedHttp, DataUnavailable, MarketData, PolicyClient, SecClient
from .indicators import add_indicators, technical_rating
from .journal import Journal
from .llm import BaseLLM, OpenRouterLLM
from .models import Fact, Thesis
from .retrieval import VectorIndex


def load_thesis(path: str | Path) -> Tuple[Thesis, dict]:
    raw = json.loads(Path(path).read_text())
    extra = {k: raw.pop(k) for k in list(raw) if k not in Thesis.__dataclass_fields__}
    return Thesis(**raw), extra


def build_live_inputs(thesis: Thesis, settings: Settings, extra: Optional[dict] = None,
                      offline: bool = False) -> Tuple[Dict[str, Fact], VectorIndex, list]:
    extra = extra or {}
    notes = []
    http = CachedHttp(settings.cache_dir, user_agent=settings.sec_user_agent, offline=offline)
    facts: Dict[str, Fact] = {}
    index = VectorIndex(settings.retrieval.min_score, settings.retrieval.candidates, settings.retrieval.top_k)

    # Prices -> technical facts (exact values from code)
    try:
        px = add_indicators(MarketData(http).daily(thesis.ticker, start="2023-01-01"))
        last = px.iloc[-1]
        as_of = str(last["date"])
        facts["px_close"] = Fact("px_close", round(float(last["close"]), 2), "daily prices", 1, as_of)
        facts["px_atr14"] = Fact("px_atr14", round(float(last["atr14"]), 2), "daily prices", 1, as_of)
        facts["px_technical"] = Fact("px_technical", technical_rating(last), "daily prices", 1, as_of)
    except DataUnavailable as exc:
        notes.append(f"prices unavailable: {exc}")

    # SEC figures and filing text
    try:
        sec = SecClient(http)
        facts.update(sec.quarterly_facts(thesis.ticker))
        for filing in sec.recent_filings(thesis.ticker, forms=("10-Q", "10-K"), limit=2) + \
                sec.recent_filings(thesis.ticker, forms=("8-K",), limit=3):
            index.add(sec.filing_chunks(filing))
    except DataUnavailable as exc:
        notes.append(f"SEC data unavailable: {exc}")

    # Policy status and text
    if thesis.policy_topic:
        try:
            policy = PolicyClient(http)
            status = policy.status_fact(thesis.policy_topic)
            if status:
                facts["policy_status"] = status
            index.add(policy.chunks(thesis.policy_topic, thesis.ticker))
        except DataUnavailable as exc:
            notes.append(f"policy data unavailable: {exc}")
        if extra.get("policy_direction") in ("favorable", "unfavorable") and "policy_status" in facts:
            # Direction is a human judgment recorded with the thesis, not inferred by a model.
            facts["policy_direction"] = Fact("policy_direction", extra["policy_direction"], "thesis owner", 2,
                                             facts["policy_status"].as_of)
    return facts, index, notes


def run_live_review(thesis_path: str, trigger: str, settings: Settings, mode: str = "tot",
                    llm: Optional[BaseLLM] = None, journal: Optional[Journal] = None,
                    market_shock: bool = False, offline: bool = False) -> Tuple[ReviewResult, list]:
    thesis, extra = load_thesis(thesis_path)
    facts, index, notes = build_live_inputs(thesis, settings, extra, offline=offline)
    llm = llm or OpenRouterLLM(api_key=settings.openrouter_api_key)
    reviewer = ThesisReviewer(index, llm, settings, researcher_llm=llm, journal=journal)
    result = reviewer.review(thesis, facts, trigger, today=date.today(), market_shock=market_shock,
                             prior_misses=int(extra.get("prior_misses", 0)),
                             last_review=extra.get("last_review"), mode=mode)
    return result, notes
