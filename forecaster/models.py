"""Shared-state contract (Checkpoint 5).

Agents never pass free-form messages to each other. They read and write
these records, and every record validates its required fields. A record that
breaks the contract raises ContractError, which the controller turns into a
retry and then the failover path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

TIERS = {1: "official", 2: "company", 3: "news", 4: "unverified"}
PILLARS = ("technical", "fundamental", "policy")
RATINGS = ("supportive", "weakening", "broken")
ACTIONS = ("hold", "add", "trim", "exit", "open")
CONVICTION_ORDER = ("exit_review", "low", "medium", "high")


class ContractError(ValueError):
    """Raised when a record is missing a required field or has a bad value."""


@dataclass
class Fact:
    """An exact, structured value from a Tier 1 or Tier 2 source."""

    key: str
    value: float | str
    source: str
    tier: int
    as_of: str  # ISO date

    def validate(self) -> "Fact":
        if self.tier not in TIERS:
            raise ContractError(f"fact {self.key}: bad tier {self.tier}")
        if not self.source:
            raise ContractError(f"fact {self.key}: missing source")
        _check_date(self.as_of, f"fact {self.key}")
        return self


@dataclass
class Chunk:
    """A retrievable passage of long text with the metadata filters need."""

    chunk_id: str
    ticker: str
    form: str
    section: str
    filed: str
    tier: int
    source_url: str
    text: str
    superseded: bool = False

    def validate(self) -> "Chunk":
        for name in ("chunk_id", "ticker", "form", "section", "source_url", "text"):
            if not getattr(self, name):
                raise ContractError(f"chunk missing {name}")
        _check_date(self.filed, f"chunk {self.chunk_id}")
        if self.tier not in TIERS:
            raise ContractError(f"chunk {self.chunk_id}: bad tier")
        return self


@dataclass
class Evidence:
    """A retrieved passage as passed between agents. Source, tier and date are required."""

    chunk_id: str
    text: str
    source_url: str
    tier: int
    filed: str
    section: str
    score: float
    flagged_injection: bool = False
    label: str = ""  # supports | contradicts | irrelevant (set by the Reviewer)

    def validate(self) -> "Evidence":
        if not (self.chunk_id and self.source_url and self.filed):
            raise ContractError("evidence missing source, id or date")
        if self.tier not in TIERS:
            raise ContractError("evidence has an unknown tier")
        _check_date(self.filed, f"evidence {self.chunk_id}")
        return self


@dataclass
class Condition:
    """A testable claim an explanation depends on, checked against facts by code."""

    fact: str
    op: str  # > < >= <= == !=
    value: float | str

    def validate(self) -> "Condition":
        if self.op not in (">", "<", ">=", "<=", "==", "!="):
            raise ContractError(f"condition on {self.fact}: bad operator {self.op}")
        return self


@dataclass
class Branch:
    """One node of the Tree of Thoughts: a candidate explanation of the conflict."""

    branch_id: str
    depth: int
    explanation: str
    implied_action: str
    conditions: List[Condition] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    pillar_effects: Dict[str, str] = field(default_factory=dict)
    settling_event: str = ""
    settling_date: str = ""
    parent_id: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    coverage: int = 0
    testability: int = 0
    reviewer_confidence: float = 0.0
    challenge: str = ""
    score: float = 0.0
    status: str = "open"  # open | pruned | set_aside | expanded | selected | evidence_pending
    prune_reason: str = ""

    def validate(self) -> "Branch":
        if not self.branch_id or not self.explanation:
            raise ContractError("branch missing id or explanation")
        if self.implied_action not in ACTIONS:
            raise ContractError(f"branch {self.branch_id}: action '{self.implied_action}' not allowed")
        for pillar, rating in self.pillar_effects.items():
            if pillar not in PILLARS or rating not in RATINGS:
                raise ContractError(f"branch {self.branch_id}: bad pillar effect {pillar}={rating}")
        for c in self.conditions:
            c.validate()
        for e in self.evidence:
            e.validate()
        if self.settling_date:
            _check_date(self.settling_date, f"branch {self.branch_id}")
        return self

    def tier1_support(self) -> int:
        return sum(1 for e in self.evidence if e.label == "supports" and e.tier == 1)


@dataclass
class Thesis:
    """A long-term holding under review."""

    ticker: str
    company: str
    sector: str
    horizon_months: int
    summary: str
    checkpoints: Dict[str, float]
    position_pct: float
    stop_price: float
    conviction: str = "high"
    policy_topic: str = ""


@dataclass
class Recommendation:
    """The only thing that leaves the agent system. Output contract (Checkpoint 6)."""

    ticker: str
    action: str
    size_change_pct: float
    stop_price: float
    conviction_before: str
    conviction_after: str
    explanation: str
    evidence_ids: List[str]
    falsifier: str
    runner_up: str = ""
    contested: bool = False
    route: str = ""
    route_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    fallback: bool = False

    def validate(self) -> "Recommendation":
        if self.action not in ACTIONS:
            raise ContractError(f"recommendation action '{self.action}' not allowed")
        if not self.fallback:
            if not self.evidence_ids:
                raise ContractError("recommendation must cite evidence")
            if not self.falsifier:
                raise ContractError("recommendation must say what would change the view")
        if self.stop_price <= 0:
            raise ContractError("recommendation must keep a stop-loss")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _check_date(value: str, where: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{where}: bad or missing date '{value}'") from exc


def condition_holds(cond: Condition, facts: Dict[str, Fact]) -> Optional[bool]:
    """True/False if the fact exists, None if it is unknown (evidence pending)."""
    fact = facts.get(cond.fact)
    if fact is None:
        return None
    left, right = fact.value, cond.value
    try:
        left, right = float(left), float(right)
    except (TypeError, ValueError):
        left, right = str(left), str(right)
    ops = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    try:
        return ops[cond.op](left, right)
    except TypeError:
        return None
