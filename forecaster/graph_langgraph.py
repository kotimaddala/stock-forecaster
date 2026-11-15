"""Optional LangGraph wiring of the same review workflow (requires `pip install langgraph`).

The core controller (controller.py) is plain Python so it runs and is tested
anywhere. This module expresses the same coordination as an explicit
LangGraph state graph, matching the Checkpoint 5 architecture diagram:

  inputs -> (failover | review) -> route -> (auto | notify | human | blocked) -> journal

Each node is a thin function over the shared state; the ToT search itself
runs inside the `review` node through ThesisReviewer.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, TypedDict

from .controller import ThesisReviewer
from .guardrails import check_facts


class ReviewState(TypedDict, total=False):
    thesis: Any
    facts: Dict[str, Any]
    trigger: str
    today: date
    market_shock: bool
    input_ok: bool
    result: Any
    route: str
    log: list


def build_graph(reviewer: ThesisReviewer):
    from langgraph.graph import END, START, StateGraph  # imported lazily: optional dependency

    def inputs(state: ReviewState) -> ReviewState:
        required = ["revenue_growth_yoy", "px_technical", "px_close"]
        report = check_facts(state["facts"], required, state["today"], reviewer.s.max_data_age_days)
        state["input_ok"] = report.ok
        state.setdefault("log", []).append(f"inputs ok={report.ok} {report.violations}")
        return state

    def review(state: ReviewState) -> ReviewState:
        state["result"] = reviewer.review(state["thesis"], state["facts"], state["trigger"], state["today"],
                                          market_shock=state.get("market_shock", False))
        state["route"] = state["result"].recommendation.route
        return state

    def failover(state: ReviewState) -> ReviewState:
        state["result"] = reviewer.review(state["thesis"], state["facts"], state["trigger"], state["today"])
        state["route"] = "notify"
        state["log"].append("failover: insufficient evidence; conviction unchanged")
        return state

    def human(state: ReviewState) -> ReviewState:
        state["log"].append("queued for human approval (reviewer card)")
        return state

    def notify(state: ReviewState) -> ReviewState:
        state["log"].append("applied automatically; owner notified")
        return state

    def auto(state: ReviewState) -> ReviewState:
        state["log"].append("logged; no position change")
        return state

    def blocked(state: ReviewState) -> ReviewState:
        state["log"].append("blocked by guardrails; no approval path")
        return state

    g = StateGraph(ReviewState)
    for name, fn in [("inputs", inputs), ("review", review), ("failover", failover), ("human", human),
                     ("notify", notify), ("auto", auto), ("blocked", blocked)]:
        g.add_node(name, fn)
    g.add_edge(START, "inputs")
    g.add_conditional_edges("inputs", lambda s: "review" if s["input_ok"] else "failover",
                            {"review": "review", "failover": "failover"})
    g.add_conditional_edges("review", lambda s: s["route"],
                            {"human": "human", "notify": "notify", "auto": "auto", "blocked": "blocked"})
    for terminal in ("failover", "human", "notify", "auto", "blocked"):
        g.add_edge(terminal, END)
    return g.compile()
