"""Deterministic decision rules: pillars, conviction, and risk-based routing.

These encode the Checkpoint 1.1 conviction table and the Checkpoint 6 routing
tiers. No model can override them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import RoutingSettings
from .models import Evidence, Fact, Thesis

LEVELS = ["exit_review", "low", "medium", "high"]


def fundamental_rating(facts: Dict[str, Fact], thesis: Thesis, prior_misses: int = 0) -> str:
    """Checkpoint 1.1: one missed checkpoint while still growing = weakening;
    two consecutive misses or shrinking revenue = broken."""
    growth = facts.get("revenue_growth_yoy")
    target = thesis.checkpoints.get("revenue_growth_min")
    if growth is None or target is None:
        return "weakening"  # unknown is never treated as supportive
    g = float(growth.value)
    margin_change = facts.get("gross_margin_change")
    if g < 0 or (g < target and prior_misses >= 1):
        return "broken"
    if g < target or (margin_change is not None and float(margin_change.value) < -0.02):
        return "weakening"
    return "supportive"


def policy_rating(facts: Dict[str, Fact], thesis: Thesis) -> str:
    """Stage matters: proposed changes can only weaken; enacted ones can break or restore."""
    status = facts.get("policy_status")
    direction = facts.get("policy_direction")  # "favorable" | "unfavorable"
    if not thesis.policy_topic:
        return "supportive"
    if status is None:
        return "weakening"
    if direction is not None and direction.value == "unfavorable":
        return "broken" if status.value in ("final", "enacted") else "weakening"
    return "supportive"


def conviction_from_pillars(pillars: Dict[str, str], market_shock: bool = False) -> str:
    """Checkpoint 1.1 table. A market shock caps conviction at medium."""
    ratings = list(pillars.values())
    if "broken" in ratings:
        level = "exit_review"
    else:
        weak = ratings.count("weakening")
        level = "high" if weak == 0 else "medium" if weak == 1 else "low"
    if market_shock and level == "high":
        level = "medium"
    return level


def apply_upgrade_rule(previous: str, proposed: str, confirming_reviews: int) -> str:
    """Quick to downgrade, slow to upgrade: rising needs two consecutive confirmations."""
    if LEVELS.index(proposed) <= LEVELS.index(previous):
        return proposed
    if confirming_reviews >= 2:
        return LEVELS[min(LEVELS.index(previous) + 1, LEVELS.index(proposed))]
    return previous


def evidence_can_break(evidence: List[Evidence]) -> bool:
    """Only official (Tier 1) supporting evidence may move a pillar to broken."""
    return any(e.label == "supports" and e.tier == 1 for e in evidence)


@dataclass
class RouteDecision:
    route: str  # auto | notify | human | blocked
    score: int
    reasons: List[str] = field(default_factory=list)


def route_action(action: str, size_change_pct: float, conviction_before: str, conviction_after: str,
                 contested: bool, reviewer_confidence: float, market_shock: bool,
                 integrity_flags: int, guardrail_violations: int, rule_change: bool,
                 settings: RoutingSettings) -> RouteDecision:
    """Checkpoint 6 risk scoring. Any single hard trigger sends the case to the human."""
    reasons: List[str] = []
    if guardrail_violations:
        return RouteDecision("blocked", 99, [f"{guardrail_violations} guardrail violation(s)"])
    score = 0
    if action in ("open", "exit", "add"):
        reasons.append(f"high-impact action: {action}")
        score += 3
    if action == "trim" and abs(size_change_pct) >= settings.large_trim_pct_of_account:
        reasons.append(f"large trim ({abs(size_change_pct):.1f}% of account)")
        score += 3
    if contested:
        reasons.append("explanations still contested after review")
        score += 3
    if reviewer_confidence < settings.min_reviewer_confidence:
        reasons.append(f"low reviewer confidence ({reviewer_confidence:.2f})")
        score += 3
    jump = abs(LEVELS.index(conviction_after) - LEVELS.index(conviction_before))
    if jump >= 2:
        reasons.append("conviction changed by two or more levels")
        score += 3
    if market_shock:
        reasons.append("market shock in effect")
        score += 3
    if integrity_flags:
        reasons.append("possible prompt injection in retrieved text")
        score += 3
    if rule_change:
        reasons.append("proposes a rule or limit change")
        score += 3
    if score >= settings.human_score_threshold:
        return RouteDecision("human", score, reasons)
    if action == "trim" or jump == 1:
        return RouteDecision("notify", score, reasons or ["small rule-based change"])
    return RouteDecision("auto", score, reasons or ["no change in position"])
