"""Turns scenario dictionaries into live objects and checks results against expectations."""

from __future__ import annotations

import copy
import dataclasses
from datetime import date
from typing import Dict, List, Optional, Tuple

from forecaster.config import Settings
from forecaster.controller import ReviewResult, ThesisReviewer
from forecaster.llm import BaseLLM, ScriptedLLM
from forecaster.models import Chunk, Fact, Thesis
from forecaster.retrieval import VectorIndex


def build(scn: dict, settings: Optional[Settings] = None, llm: Optional[BaseLLM] = None,
          shuffle_seed: Optional[int] = None) -> Tuple[ThesisReviewer, Thesis, Dict[str, Fact]]:
    settings = settings or Settings(openrouter_api_key="")
    if scn.get("search"):
        settings = dataclasses.replace(settings, search=dataclasses.replace(settings.search, **scn["search"]))
    chunks = [Chunk(**c) for c in copy.deepcopy(scn["chunks"])]
    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(chunks)
    index = VectorIndex(settings.retrieval.min_score, settings.retrieval.candidates, settings.retrieval.top_k)
    index.add(chunks)
    facts = {k: Fact(k, v[0], v[1], v[2], v[3]).validate() for k, v in scn["facts"].items()}
    thesis = Thesis(**copy.deepcopy(scn["thesis"]))
    llm = llm or ScriptedLLM(script=copy.deepcopy(scn["script"]))
    return ThesisReviewer(index, llm, settings), thesis, facts


def run(scn: dict, mode: str = "tot", llm: Optional[BaseLLM] = None, settings: Optional[Settings] = None,
        shuffle_seed: Optional[int] = None) -> ReviewResult:
    reviewer, thesis, facts = build(scn, settings, llm, shuffle_seed)
    return reviewer.review(thesis, facts, scn["trigger"], today=date.fromisoformat(scn["today"]),
                           market_shock=scn["market_shock"], last_review=scn["last_review"], mode=mode)


def check(scn: dict, result: ReviewResult) -> List[str]:
    """Return a list of failed expectations (empty list = pass)."""
    exp, rec = scn["expected"], result.recommendation
    failures = []
    branches = {b.branch_id: b for b in result.branches}

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    if "action" in exp:
        expect(rec.action == exp["action"], f"action {rec.action} != {exp['action']}")
    if "route" in exp:
        expect(rec.route == exp["route"], f"route {rec.route} != {exp['route']}")
    if "fallback" in exp:
        expect(rec.fallback == exp["fallback"], f"fallback {rec.fallback} != {exp['fallback']}")
    if "contested" in exp:
        expect(rec.contested == exp["contested"], f"contested {rec.contested} != {exp['contested']}")
    if "llm_calls" in exp:
        expect(result.llm_calls == exp["llm_calls"], f"llm calls {result.llm_calls} != {exp['llm_calls']}")
    if "max_llm_calls_le" in exp:
        expect(result.llm_calls <= exp["max_llm_calls_le"], f"llm calls {result.llm_calls} > {exp['max_llm_calls_le']}")
    if "conviction_after" in exp:
        expect(rec.conviction_after == exp["conviction_after"], f"conviction {rec.conviction_after}")
    if "incomplete" in exp:
        expect(result.incomplete == exp["incomplete"], f"incomplete {result.incomplete}")
    if "selected" in exp:
        sel = [b.branch_id for b in result.branches if b.status == "selected"]
        expect(sel == [exp["selected"]], f"selected {sel} != {exp['selected']}")
    for bid in exp.get("pruned", []):
        expect(bid in branches and branches[bid].status == "pruned", f"{bid} not pruned")
    for bid, status in exp.get("status", {}).items():
        got = branches.get(bid).status if bid in branches else None
        expect(got == status, f"{bid} status {got} != {status}")
    if exp.get("injection_flagged"):
        flagged = any(e.flagged_injection for b in result.branches for e in b.evidence)
        expect(flagged, "injection not flagged")
        expect(any("injection" in r for r in rec.route_reasons), "injection did not trigger human review")
    for bad in exp.get("action_not", []):
        expect(rec.action != bad, f"action must not be {bad}")
    if "reason_contains" in exp:
        expect(any(exp["reason_contains"] in r for r in rec.route_reasons), f"missing reason {exp['reason_contains']}")
    if "refine_calls" in exp:
        refine = sum(1 for b in result.branches if b.depth == 2)
        expect(refine == exp["refine_calls"], f"depth-2 nodes {refine} != {exp['refine_calls']}")
    return failures


def isolation_violations(scn: dict, result: ReviewResult) -> List[str]:
    """Evidence from another company or a superseded filing must never be used."""
    by_id = {c["chunk_id"]: c for c in scn["chunks"]}
    bad = []
    for b in result.branches:
        for e in b.evidence:
            c = by_id.get(e.chunk_id, {})
            if c.get("ticker") != scn["thesis"]["ticker"]:
                bad.append(f"{e.chunk_id}: other company")
            if c.get("chunk_id") == "G-MDA-OLD":
                bad.append(f"{e.chunk_id}: superseded filing")
    return bad
