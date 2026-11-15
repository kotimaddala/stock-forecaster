"""End-to-end test of the live pipeline using cached copies of each public API response.

The responses mimic the real formats of SEC EDGAR (company_tickers.json,
submissions, companyfacts, filing HTML), Stooq CSV and the Federal Register
API, so parsing and wiring are tested without network access.
"""

import json
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from forecaster.config import Settings
from forecaster.data.http import CachedHttp
from forecaster.data.market import STOOQ_URL
from forecaster.data.policy import FEDREG_URL
from forecaster.data.sec import FACTS_URL, SUBMISSIONS_URL, TICKERS_URL
from forecaster.journal import Journal
from forecaster.llm import ScriptedLLM
from forecaster.pipeline import build_live_inputs, load_thesis, run_live_review

CIK = 123456


def _put(http: CachedHttp, url, body, params=None):
    path = http._path(url, params)
    path.write_text(body if isinstance(body, str) else json.dumps(body))


def _seed_cache(cache: Path, today: date):
    http = CachedHttp(cache, offline=True)
    _put(http, TICKERS_URL, {"0": {"cik_str": CIK, "ticker": "SYNX", "title": "Synthetic Corp"}})
    _put(http, SUBMISSIONS_URL.format(cik=CIK), {"filings": {"recent": {
        "form": ["10-Q", "8-K"], "filingDate": ["2026-10-30", "2026-10-30"],
        "accessionNumber": ["0000123456-26-000010", "0000123456-26-000011"],
        "primaryDocument": ["synx-10q.htm", "synx-8k.htm"]}}})
    rows = []
    for year, val, gp in ((2025, 100.0, 31.0), (2026, 114.0, 31.35)):
        for q, (s, e) in enumerate([("04-01", "06-30"), ("07-01", "09-30")]):
            rows.append(("Revenues", f"{year}-{s}", f"{year}-{e}", val + q, f"{year}-10-30"))
            rows.append(("GrossProfit", f"{year}-{s}", f"{year}-{e}", gp + q * (0 if year == 2025 else -3),
                         f"{year}-10-30"))
    rows.append(("Revenues", "2025-01-01", "2025-03-31", 98.0, "2025-04-30"))
    facts = {"facts": {"us-gaap": {}}}
    for concept, s, e, v, f in rows:
        facts["facts"]["us-gaap"].setdefault(concept, {"units": {"USD": []}})["units"]["USD"].append(
            {"start": s, "end": e, "val": v, "form": "10-Q", "filed": f})
    _put(http, FACTS_URL.format(cik=CIK), facts)
    body = ("revenue increased because demand from utilities stayed strong while steel supply was tight " * 20)
    html = (f"<html><body><p>Item 2. Management's Discussion</p><p>Item 1A. Risk Factors</p>"
            f"<p>Item 2. Management's Discussion and Analysis</p><p>{body}</p>"
            f"<table><tr><td>999</td></tr></table>"
            f"<p>Item 1A. Risk Factors</p><p>{'a shortage of electrical steel could hurt margins ' * 15}</p></body></html>")
    _put(http, f"https://www.sec.gov/Archives/edgar/data/{CIK}/000012345626000010/synx-10q.htm", html)
    _put(http, f"https://www.sec.gov/Archives/edgar/data/{CIK}/000012345626000011/synx-8k.htm",
         "<html><body>" + "Synthetic Corp announced a new utility contract for transformers. " * 15 + "</body></html>")
    days = [today - timedelta(days=i) for i in range(300, 0, -1)]
    closes = np.linspace(40, 60, len(days))
    csv = "Date,Open,High,Low,Close,Volume\n" + "\n".join(
        f"{d},{c - .2:.2f},{c + .5:.2f},{c - .5:.2f},{c:.2f},1000000" for d, c in zip(days, closes))
    _put(http, STOOQ_URL, csv, {"s": "synx.us", "i": "d"})
    fr_params = {"conditions[term]": "grid program", "order": "newest", "per_page": 1,
                 "fields[]": ["title", "type", "abstract", "publication_date", "html_url", "agencies"]}
    doc = {"results": [{"title": "Grid program disbursement schedule", "type": "Proposed Rule",
                        "abstract": "The Department proposes deferring part of the grid program funding to a later year.",
                        "publication_date": "2026-11-03", "html_url": "https://www.federalregister.gov/d/synthetic"}]}
    _put(http, FEDREG_URL, doc, fr_params)
    _put(http, FEDREG_URL, doc, dict(fr_params, per_page=5))


def _thesis(tmp_path):
    p = tmp_path / "thesis.json"
    p.write_text(json.dumps({"ticker": "SYNX", "company": "Synthetic Corp", "sector": "Industrials",
                             "horizon_months": 12, "summary": "s", "checkpoints": {"revenue_growth_min": 0.2},
                             "position_pct": 6.0, "stop_price": 41.0, "policy_topic": "grid program",
                             "policy_direction": "unfavorable"}))
    return p


def test_build_live_inputs_from_cached_public_formats(tmp_path):
    today = date.today()
    settings = Settings(data_dir=tmp_path)
    _seed_cache(settings.cache_dir, today)
    thesis, extra = load_thesis(_thesis(tmp_path))
    facts, index, notes = build_live_inputs(thesis, settings, extra, offline=True)
    assert notes == []
    assert facts["revenue_growth_yoy"].value == round(115 / 101 - 1, 4)   # latest quarter vs same quarter last year
    assert facts["px_technical"].value == "supportive"
    assert facts["policy_status"].value == "proposed"
    assert facts["policy_direction"].value == "unfavorable" and facts["policy_direction"].tier == 2
    sections = {c.section for c in index.chunks}
    assert {"Management's Discussion and Analysis", "Risk Factors", "Current report"} <= sections
    assert not any("999" in c.text for c in index.chunks)               # tables excluded from text
    assert all(c.tier == 1 and c.source_url.startswith("https://") for c in index.chunks)


def test_run_live_review_end_to_end_with_scripted_models(tmp_path):
    settings = Settings(data_dir=tmp_path)
    _seed_cache(settings.cache_dir, date.today())
    llm = ScriptedLLM(script={
        "analyst.depth1": {"branches": [
            {"id": "A", "explanation": "Funding deferral delays revenue.", "implied_action": "hold",
             "queries": ["deferring grid program funding"], "pillar_effects": {"policy": "weakening"}},
            {"id": "B", "explanation": "Steel shortage hurts margins.", "implied_action": "trim",
             "queries": ["shortage of electrical steel margins"], "pillar_effects": {"fundamental": "weakening"}}]},
        "reviewer.A": {"labels": {}, "coverage": 1, "testability": 1, "confidence": 0.8},
        "reviewer.B": {"labels": {}, "coverage": 2, "testability": 2, "confidence": 0.8},
        "analyst.refine.A": {"refinements": []}, "analyst.refine.B": {"refinements": []},
        "analyst.recommend": {"explanation": "Margins are under pressure from steel supply.", "falsifier": "x",
                              "evidence_ids": []},
    })
    # Label whatever was retrieved as supporting B, as a real reviewer would after reading it.
    original = llm.complete_json

    def complete(tag, model, system, user):
        out = original(tag, model, system, user)
        if tag == "reviewer.B":
            out["labels"] = {cid.split()[0]: "supports" for cid in user.split("id=")[1:]}
        return out
    llm.complete_json = complete
    journal = Journal(tmp_path / "j.sqlite")
    result, notes = run_live_review(str(_thesis(tmp_path)), "10-Q filed", settings, llm=llm, journal=journal,
                                    offline=True)
    assert notes == []
    assert result.recommendation.action in ("trim", "hold")
    assert result.recommendation.route in ("human", "notify", "auto")
    assert journal.summary()["reviews"] == 1


def test_langgraph_wiring_with_minimal_stub(monkeypatch, tmp_path):
    """Checks the graph's routing logic using a tiny stand-in for the langgraph API."""
    class StubGraph:
        def __init__(self, _):
            self.nodes, self.edges, self.cond = {}, {}, {}

        def add_node(self, n, f):
            self.nodes[n] = f

        def add_edge(self, a, b):
            self.edges[a] = b

        def add_conditional_edges(self, a, fn, mapping):
            self.cond[a] = (fn, mapping)

        def compile(self):
            return self

        def invoke(self, state):
            node = self.edges["START"]
            while node != "END":
                state = self.nodes[node](state)
                node = self.cond[node][1][self.cond[node][0](state)] if node in self.cond else self.edges[node]
            return state

    mod = types.ModuleType("langgraph.graph")
    mod.StateGraph, mod.START, mod.END = StubGraph, "START", "END"
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", mod)
    from eval.harness import build
    from eval.scenario_data import all_scenarios
    from forecaster.graph_langgraph import build_graph
    scn = all_scenarios()[0]
    reviewer, thesis, facts = build(scn)
    out = build_graph(reviewer).invoke({"thesis": thesis, "facts": facts, "trigger": scn["trigger"],
                                        "today": date.fromisoformat(scn["today"])})
    assert out["route"] == "human" and "queued for human approval" in out["log"][-1]
