"""Tests for retrieval, SEC parsing helpers, the scanner, the broker gate, the journal and the LLM parser."""

import numpy as np
import pandas as pd
import pytest

from eval.scenario_data import BASE_CHUNKS, T
from forecaster.broker import DryRunBroker, Order, OrderRejected
from forecaster.config import RiskLimits
from forecaster.data.sec import _match_prior_year, _quarterly_series, chunk_words_with_overlap, split_sections
from forecaster.journal import Journal
from forecaster.llm import LLMError, ScriptedLLM, parse_json
from forecaster.models import Chunk
from forecaster.retrieval import VectorIndex
from forecaster.scanner import scan


@pytest.fixture
def index():
    idx = VectorIndex(min_score=0.08)
    idx.add([Chunk(**c) for c in BASE_CHUNKS])
    return idx


def test_filters_exclude_other_company_and_superseded(index):
    hits = index.search("demand from utility customers weakened shortage", ticker=T)
    ids = {e.chunk_id for e in hits}
    assert "V-MDA-1" not in ids and "G-MDA-OLD" not in ids
    unfiltered = {e.chunk_id for e in index.search("demand from utility customers weakened", ticker="*",
                                                    include_superseded=True)}
    assert "V-MDA-1" in unfiltered  # proves the filter is what prevents the leak


def test_threshold_returns_nothing_for_unrelated_query(index):
    assert index.search("zzqx quantum banana", ticker=T) == []
    assert index.logs[-1].nothing_relevant


def test_every_result_carries_source_tier_and_date(index):
    for e in index.search("gross margin declined", ticker=T):
        assert e.source_url and e.filed and e.tier in (1, 2, 3, 4)


def test_quarterly_series_and_prior_year():
    gaap = {"Revenues": {"units": {"USD": [
        {"start": "2025-07-01", "end": "2025-09-30", "val": 100, "form": "10-Q", "filed": "2025-10-30"},
        {"start": "2025-01-01", "end": "2025-09-30", "val": 300, "form": "10-Q", "filed": "2025-10-30"},  # 9 months
        {"start": "2026-07-01", "end": "2026-09-30", "val": 114, "form": "10-Q", "filed": "2026-10-30"},
    ]}}}
    series = _quarterly_series(gaap, ("Revenues",))
    assert [s[1] for s in series] == [100, 114]
    assert _match_prior_year(series, "2026-09-30")[1] == 100


def test_section_split_prefers_real_section_over_table_of_contents():
    text = ("Item 2. Management's Discussion\nItem 1A. Risk Factors\n"
            "Item 2. Management's Discussion and Analysis\n" + "revenue grew strongly " * 40 +
            "\nItem 1A. Risk Factors\n" + "suppliers may fail " * 30)
    sections = split_sections(text, "10-Q")
    assert "revenue grew" in sections["Management's Discussion and Analysis"]
    assert "suppliers" in sections["Risk Factors"]


def test_chunk_overlap():
    pieces = chunk_words_with_overlap(" ".join(str(i) for i in range(1000)), 375, 56)
    assert len(pieces) >= 3
    assert pieces[0].split()[-56:] == pieces[1].split()[:56]


def _prices(breakout: bool) -> pd.DataFrame:
    n = 220
    close = np.linspace(40, 50, n)
    vol = np.full(n, 2_000_000.0)
    if breakout:
        close[-1] = 54
        vol[-1] = 7_000_000
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n).date, "open": close - 0.3,
                       "high": close + 0.5, "low": close - 0.5, "close": close, "volume": vol})
    return df


def test_scanner_finds_breakout_with_full_plan():
    ideas = scan({"SYN-A": _prices(True), "SYN-B": _prices(False)}, 100_000, RiskLimits())
    assert [i.ticker for i in ideas] == ["SYN-A"]
    idea = ideas[0]
    assert idea.stop < idea.entry < idea.target and idea.route == "human"
    assert idea.risk_pct <= 1.0 and idea.position_pct <= 10.0


def test_scanner_halves_size_in_market_shock():
    normal = scan({"SYN-A": _prices(True)}, 100_000, RiskLimits())[0]
    shocked = scan({"SYN-A": _prices(True)}, 100_000, RiskLimits(), market_shock=True)[0]
    assert shocked.shares < normal.shares


def test_broker_requires_approval_stop_and_no_shorts(tmp_path):
    broker = DryRunBroker(tmp_path / "orders.jsonl")
    with pytest.raises(OrderRejected):
        broker.submit(Order("SYN-A", "buy", 10, stop_price=45))                       # not approved
    with pytest.raises(OrderRejected):
        broker.submit(Order("SYN-A", "buy", 10, approved=True))                       # no stop
    with pytest.raises(OrderRejected):
        broker.submit(Order("SYN-A", "sell", 10, approved=True), held_qty=5)         # would be short
    assert broker.submit(Order("SYN-A", "buy", 10, stop_price=45, approved=True))["mode"] == "dry-run"


def test_lessons_need_all_four_tests():
    j = Journal()
    assert j.propose_lesson("breakout", "weak-sector breakouts fail", 3, 0.2, True, True, "r") == "candidate"
    assert j.propose_lesson("breakout", "weak-sector breakouts fail", 9, 0.2, False, True, "r") == "candidate"
    assert j.propose_lesson("breakout", "low-volume breakouts fail", 9, 0.25, True, True,
                            "no volume confirmation") == "validated"
    assert j.lessons_for("breakout") == ["low-volume breakouts fail"]


def test_journal_rejects_unknown_reason_codes():
    with pytest.raises(ValueError):
        Journal().record_decision(1, "approve", "because")


def test_parse_json_handles_fences_and_garbage():
    assert parse_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_scripted_llm_malformed_then_valid():
    llm = ScriptedLLM(script={"t": ["MALFORMED", {"ok": True}]})
    with pytest.raises(LLMError):
        llm.complete_json("t", "m", "s", "u")
    assert llm.complete_json("t", "m", "s", "u") == {"ok": True}


def test_unexplained_conflict_still_applies_code_pillars():
    """If no explanation is supported, code ratings still set conviction and a human sees it."""
    from forecaster.config import load_settings
    from forecaster.controller import ThesisReviewer
    from forecaster.llm import ScriptedLLM
    from forecaster.models import Thesis
    from forecaster.retrieval import VectorIndex
    thesis = Thesis("SYN-X", "X", "Industrials", 12, "s", {"revenue_growth_min": 0.08}, 6.0, 40.0, "high")
    ctl = ThesisReviewer(VectorIndex(), ScriptedLLM({}), load_settings())
    rec = ctl._unexplained(thesis, {"technical": "broken", "fundamental": "supportive", "policy": "supportive"},
                           False, 0, ["best explanation scored 0.0 (< 4.0)"])
    assert rec.action == "hold" and rec.conviction_after == "exit_review"
    assert rec.route == "human"
    rec2 = ctl._unexplained(thesis, {"technical": "weakening", "fundamental": "supportive",
                                     "policy": "supportive"}, False, 0, ["x"])
    assert rec2.conviction_after == "medium" and rec2.route in ("notify", "human")
