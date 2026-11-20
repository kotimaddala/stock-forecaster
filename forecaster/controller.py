"""The Controller: a Tree-of-Thoughts thesis review run by ordinary code (Checkpoints 4-6).

Flow for one review:
  input checks -> pillar ratings (code) -> no conflict? quick hold
  -> depth 1: Analyst brainstorms explanations
  -> code gate: prune any explanation contradicted by a Tier 1 fact
  -> Researcher retrieves evidence; Reviewer labels it and scores the branch
  -> beam search (width 2) to depth 2, with pruning, budgets and early stop
  -> one evidence-only rebuttal round if the Reviewer challenged the leader
  -> decision rules (ties, contested cases decided by action, not by truth)
  -> conviction rules -> recommendation text -> source verification -> risk routing

Every routing decision here is deterministic and unit-tested; models only
generate and judge text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .agents import Analyst, Researcher, Reviewer, context_block, parse_branch
from .config import Settings
from .guardrails import GuardrailReport, check_facts, verify_recommendation
from .llm import BaseLLM, LLMError
from .models import Branch, ContractError, Evidence, Fact, Recommendation, Thesis, condition_holds
from .retrieval import VectorIndex
from .risk import (LEVELS, apply_upgrade_rule, conviction_from_pillars, evidence_can_break,
                   fundamental_rating, policy_rating, route_action)

TIER_WEIGHT = {1: 1.5, 2: 1.0, 3: 0.5, 4: 0.0}
RATING_ORDER = {"supportive": 0, "weakening": 1, "broken": 2}


@dataclass
class ReviewResult:
    recommendation: Recommendation
    branches: List[Branch]
    pillars: Dict[str, str]
    guardrails: GuardrailReport
    mode: str
    llm_calls: int = 0
    retrieval_calls: int = 0
    seconds: float = 0.0
    incomplete: bool = False
    trace: List[str] = field(default_factory=list)
    retrieval_logs: List[dict] = field(default_factory=list)


def score_branch(branch: Branch, last_review: Optional[str]) -> float:
    """Checkpoint 4 rubric: evidence strength 0-4, coverage 0-3, testability 0-2, recency 0-1."""
    sup = sum(TIER_WEIGHT.get(e.tier, 0) for e in branch.evidence if e.label == "supports")
    con = sum(TIER_WEIGHT.get(e.tier, 0) for e in branch.evidence if e.label == "contradicts")
    strength = max(0.0, min(4.0, sup - con))
    recency = 1.0 if any(e.label == "supports" and (last_review is None or e.filed > last_review)
                         for e in branch.evidence) else 0.0
    branch.score = round(strength + branch.coverage + branch.testability + recency, 2)
    return branch.score


class ThesisReviewer:
    def __init__(self, index: VectorIndex, llm: BaseLLM, settings: Settings,
                 researcher_llm: Optional[BaseLLM] = None, journal=None) -> None:
        self.s = settings
        self.index = index
        self.llm = llm
        self.analyst = Analyst(llm, settings)
        self.reviewer = Reviewer(llm, settings)
        self.researcher = Researcher(index, researcher_llm, settings)
        self.journal = journal

    # ------------------------------------------------------------------ API
    def review(self, thesis: Thesis, facts: Dict[str, Fact], trigger: str, today: date,
               market_shock: bool = False, prior_misses: int = 0, last_review: Optional[str] = None,
               confirming_reviews: int = 0, mode: str = "tot") -> ReviewResult:
        started = time.time()
        self._llm0, self._ret0 = self.llm.n_calls, self.researcher.calls
        self._log0 = len(self.index.logs)
        self.incomplete = False
        self._ticker = thesis.ticker
        trace: List[str] = []

        required = ["revenue_growth_yoy", "px_technical", "px_close"] + (["policy_status"] if thesis.policy_topic else [])
        check = check_facts(facts, required, today, self.s.max_data_age_days)
        if not check.ok:
            trace.append("input checks failed: " + "; ".join(check.violations))
            return self._finish(self._fallback(thesis, check.violations), [], {}, check, mode, started, trace)

        pillars = {
            "technical": str(facts["px_technical"].value),
            "fundamental": fundamental_rating(facts, thesis, prior_misses),
            "policy": policy_rating(facts, thesis),
        }
        trace.append(f"pillars from code: {pillars}")
        context = context_block(thesis, facts, pillars, trigger)

        if all(v == "supportive" for v in pillars.values()) and not market_shock:
            rec = self._no_conflict(thesis, pillars, confirming_reviews)
            report = verify_recommendation(rec, facts, {})
            self._route(rec, thesis, report, contested=False, confidence=1.0, shock=False, integrity=0)
            trace.append("no conflict between pillars; ToT not needed")
            return self._finish(rec, [], pillars, report, mode, started, trace)

        try:
            if mode == "single":
                return self._single_chain(thesis, facts, pillars, context, market_shock, last_review,
                                          confirming_reviews, started, trace)
            tree, frontier = self._search(thesis, facts, context, last_review, trace)
        except (LLMError, ContractError) as exc:
            trace.append(f"agent output failed after retry: {exc}")
            rec = self._fallback(thesis, [f"agent failure: {exc}"])
            return self._finish(rec, [], pillars, GuardrailReport(), mode, started, trace)

        if not frontier:
            rec = self._fallback(thesis, ["every explanation was contradicted or scored below the threshold"])
            return self._finish(rec, tree, pillars, GuardrailReport(), mode, started, trace)

        best, runner, contested = self._decide(frontier, context, last_review, trace)
        if best.score < self.s.search.prune_below:
            rec = self._fallback(thesis, [f"best explanation scored {best.score} (< {self.s.search.prune_below})"])
            return self._finish(rec, tree, pillars, GuardrailReport(), mode, started, trace)

        best.status = "selected"
        action, size_change, stop, notes = self._action(thesis, facts, best, runner, contested, today)
        pillars_after = self._apply_effects(pillars, best, trace)
        proposed = conviction_from_pillars(pillars_after, market_shock)
        conviction_after = apply_upgrade_rule(thesis.conviction, proposed, confirming_reviews)
        if conviction_after == "exit_review" and action not in ("exit",):
            notes.append("a pillar is broken by official evidence: exit review")
            action, size_change = "exit", -thesis.position_pct

        retrieved = {e.chunk_id: e for b in tree for e in b.evidence}
        known_numbers = _thesis_numbers(thesis, stop, size_change)
        rec = self._write_recommendation(thesis, facts, context, best, runner, action, size_change, stop,
                                         conviction_after, contested, notes, retrieved)
        report = verify_recommendation(rec, facts, retrieved, extra_numbers=known_numbers)
        if any("has no source" in v for v in report.violations) and not self._over_budget():
            trace.append("unsourced number in draft; asking the Analyst to rewrite once")
            rec = self._write_recommendation(thesis, facts, context, best, runner, action, size_change, stop,
                                             conviction_after, contested, notes + ["previous draft used an unsourced number"],
                                             retrieved)
            report = verify_recommendation(rec, facts, retrieved, extra_numbers=known_numbers)
        integrity = sum(1 for e in best.evidence if e.flagged_injection)
        confidence = best.reviewer_confidence
        self._route(rec, thesis, report, contested, confidence, market_shock, integrity)
        return self._finish(rec, tree, pillars_after, report, mode, started, trace)

    # ------------------------------------------------------------ search
    def _over_budget(self) -> bool:
        S = self.s.search
        return (self.llm.n_calls - self._llm0 >= S.max_llm_calls
                or self.researcher.calls - self._ret0 >= S.max_retrieval_calls)

    def _search(self, thesis, facts, context, last_review, trace) -> Tuple[List[Branch], List[Branch]]:
        S = self.s.search
        depth1 = self.analyst.propose(context, S.depth1_branches)
        trace.append(f"depth 1: {len(depth1)} explanations proposed")
        self._evaluate(depth1, thesis, facts, context, last_review, trace)
        beam = self._prune_and_beam(depth1, trace, keep_min=2)
        if not beam:
            return depth1, []
        tree = list(depth1)
        if len(beam) >= 2 and beam[0].score - beam[1].score >= S.early_stop_lead:
            trace.append(f"early stop: {beam[0].branch_id} leads by {beam[0].score - beam[1].score:.1f}")
            return tree, beam
        if not S.expand_depth2:
            return tree, beam
        frontier: List[Branch] = []
        for parent in beam:
            if self._over_budget():
                self.incomplete = True
                trace.append("budget reached; keeping current best without expanding")
                frontier.append(parent)
                continue
            kids = self.analyst.refine(context, parent, S.depth2_refinements)
            parent.status = "expanded"
            tree += kids
            self._evaluate(kids, thesis, facts, context, last_review, trace)
            alive = [k for k in kids if k.status != "pruned"]
            for k in alive:
                if k.score < S.prune_below and k.status != "evidence_pending":
                    k.status, k.prune_reason = "pruned", f"score {k.score} below {S.prune_below}"
            alive = [k for k in alive if k.status != "pruned"]
            frontier += alive or [parent]
        frontier.sort(key=lambda b: b.score, reverse=True)
        for b in frontier[S.beam_width:]:
            b.status = "set_aside"
        trace.append("depth 2 frontier: " + ", ".join(f"{b.branch_id}={b.score}" for b in frontier))
        return tree, frontier[: S.beam_width]

    def _evaluate(self, nodes: List[Branch], thesis, facts, context, last_review, trace) -> None:
        for b in nodes:
            for cond in b.conditions:
                if condition_holds(cond, facts) is False:
                    b.status = "pruned"
                    b.prune_reason = f"contradicted by Tier 1 fact {cond.fact}={facts[cond.fact].value}"
                    trace.append(f"{b.branch_id} pruned: {b.prune_reason}")
                    break
            if b.status == "pruned":
                continue
            if self._over_budget():
                self.incomplete = True
                b.status = "evidence_pending"
                continue
            budget_left = self.s.search.max_retrieval_calls - (self.researcher.calls - self._ret0)
            b.evidence = self.researcher.gather(thesis.ticker, b, budget_left=budget_left)
            self.reviewer.review(context, b)
            score_branch(b, last_review)
            if not b.evidence:
                b.status = "evidence_pending"  # unsupported yet is not the same as contradicted
            trace.append(f"{b.branch_id} scored {b.score} with {len(b.evidence)} passages")

    def _prune_and_beam(self, nodes: List[Branch], trace, keep_min: int) -> List[Branch]:
        S = self.s.search
        for b in nodes:
            if b.status == "open" and b.score < S.prune_below:
                b.status, b.prune_reason = "pruned", f"score {b.score} below {S.prune_below}"
        alive = [b for b in nodes if b.status in ("open", "evidence_pending")]
        if len(alive) < keep_min:  # Checkpoint 4 mitigation: keep at least two alive at depth 1
            revivable = sorted([b for b in nodes if b.prune_reason.startswith("score")],
                               key=lambda b: b.score, reverse=True)
            for b in revivable[: keep_min - len(alive)]:
                b.status, b.prune_reason = "open", ""
                alive.append(b)
                trace.append(f"{b.branch_id} kept alive to preserve two candidates")
        alive.sort(key=lambda b: b.score, reverse=True)
        for b in alive[S.beam_width:]:
            if b.status != "evidence_pending":  # pending branches stay visibly pending, not dismissed
                b.status = "set_aside"
        return alive[: S.beam_width]

    # ------------------------------------------------------------ decide
    def _decide(self, frontier: List[Branch], context, last_review, trace):
        S = self.s.search
        ranked = sorted(frontier, key=lambda b: b.score, reverse=True)
        best = ranked[0]
        if best.challenge and not self._over_budget():
            trace.append(f"reviewer challenged {best.branch_id}: {best.challenge}")
            queries = self.analyst.rebuttal(context, best, best.challenge)
            if queries:
                probe = Branch(branch_id=best.branch_id + "-rebuttal", depth=best.depth, explanation=best.explanation,
                               implied_action=best.implied_action, queries=queries)
                extra = self.researcher.gather(self._ticker, probe)
                known = {e.chunk_id for e in best.evidence}
                best.evidence += [e for e in extra if e.chunk_id not in known]
                self.reviewer.review(context, best, round_=2)
                score_branch(best, last_review)
                trace.append(f"after rebuttal round {best.branch_id} scored {best.score}")
            else:
                trace.append("rebuttal offered no new evidence; challenge stands")
            ranked = sorted(frontier, key=lambda b: b.score, reverse=True)
            best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        contested = False
        if runner and best.score - runner.score <= S.tie_margin:
            if best.tier1_support() != runner.tier1_support():
                if runner.tier1_support() > best.tier1_support():
                    best, runner = runner, best
                trace.append(f"tie broken by Tier 1 evidence count: {best.branch_id}")
            else:
                contested = True
                trace.append(f"contested: {best.branch_id} vs {runner.branch_id}")
        return best, runner, contested

    def _action(self, thesis: Thesis, facts, best: Branch, runner: Optional[Branch], contested: bool,
                today: date):
        """Decide on the action, not on which explanation is true (Checkpoint 5 feedback)."""
        notes: List[str] = []
        action = best.implied_action
        if contested and runner is not None:
            if best.implied_action == runner.implied_action:
                notes.append("both explanations imply the same action")
            else:
                dates = [b.settling_date for b in (best, runner) if b.settling_date]
                soonest = min(dates) if dates else ""
                window = today + timedelta(days=self.s.search.settle_window_days)
                if soonest and date.fromisoformat(soonest) <= window:
                    action = "hold"
                    notes.append(f"hold with a tighter stop; re-review on {soonest} when the question is settled")
                else:
                    action = "trim"
                    notes.append("partial action acceptable under both explanations; re-review at next filing")
        size_change = {"hold": 0.0, "exit": -thesis.position_pct, "trim": -round(thesis.position_pct / 3, 2),
                       "add": round(min(thesis.position_pct * 0.5, self.s.risk.max_position_pct - thesis.position_pct), 2),
                       "open": 0.0}[action]
        stop = thesis.stop_price
        if action in ("trim", "hold") and (contested or action == "trim"):
            close, atr = facts.get("px_close"), facts.get("px_atr14")
            if close is not None and atr is not None:
                tighter = round(float(close.value) - 2 * float(atr.value), 2)
                if tighter > stop:
                    stop = tighter
                    notes.append("stop tightened to two average daily ranges below the last close")
        return action, size_change, stop, notes

    def _apply_effects(self, pillars: Dict[str, str], best: Branch, trace) -> Dict[str, str]:
        """Evidence quality limits what a branch can do to a pillar (Checkpoint 2.1 tiers)."""
        out = dict(pillars)
        supported = [e for e in best.evidence if e.label == "supports"]
        best_tier = min((e.tier for e in supported), default=4)
        for pillar, effect in best.pillar_effects.items():
            if RATING_ORDER[effect] <= RATING_ORDER[out[pillar]]:
                continue  # branches can only add caution; restoring needs code-rated facts
            if best_tier >= 3:
                effect = "weakening" if best_tier == 3 else out[pillar]
            if effect == "broken" and not evidence_can_break(best.evidence):
                effect = "weakening"
            if RATING_ORDER[effect] > RATING_ORDER[out[pillar]]:
                trace.append(f"pillar {pillar}: {out[pillar]} -> {effect} (from {best.branch_id})")
                out[pillar] = effect
        return out

    # ------------------------------------------------------------ outputs
    def _write_recommendation(self, thesis, facts, context, best, runner, action, size_change, stop,
                              conviction_after, contested, notes, retrieved) -> Recommendation:
        try:
            data = self.analyst.recommend(context, best, action, notes)
            explanation = str(data.get("explanation", "")).strip()
            falsifier = str(data.get("falsifier", "")).strip()
            ids = [i for i in data.get("evidence_ids", []) if i in retrieved]
        except LLMError:
            explanation, falsifier, ids = "", "", []
        if not explanation:
            explanation = best.explanation
        if not falsifier:
            falsifier = (f"{best.settling_event} ({best.settling_date})" if best.settling_event
                         else "next official filing contradicts this explanation")
        if not ids:
            ids = [e.chunk_id for e in best.evidence if e.label == "supports"][:3]
        return Recommendation(
            ticker=thesis.ticker, action=action, size_change_pct=size_change, stop_price=stop,
            conviction_before=thesis.conviction, conviction_after=conviction_after,
            explanation=explanation, evidence_ids=ids, falsifier=falsifier,
            runner_up=runner.explanation if runner else "", contested=contested, notes=list(notes),
        )

    def _route(self, rec: Recommendation, thesis: Thesis, report: GuardrailReport, contested: bool,
               confidence: float, shock: bool, integrity: int) -> None:
        decision = route_action(rec.action, rec.size_change_pct, rec.conviction_before, rec.conviction_after,
                                contested, confidence, shock, integrity, len(report.violations), False,
                                self.s.routing)
        rec.route, rec.route_reasons = decision.route, decision.reasons

    def _fallback(self, thesis: Thesis, reasons: List[str]) -> Recommendation:
        rec = Recommendation(
            ticker=thesis.ticker, action="hold", size_change_pct=0.0, stop_price=thesis.stop_price,
            conviction_before=thesis.conviction, conviction_after=thesis.conviction,
            explanation="Insufficient evidence; conviction unchanged.", evidence_ids=[], falsifier="",
            notes=reasons, fallback=True, route="notify", route_reasons=["safe fallback: " + "; ".join(reasons)],
        )
        return rec.validate()

    def _no_conflict(self, thesis: Thesis, pillars, confirming_reviews: int) -> Recommendation:
        after = apply_upgrade_rule(thesis.conviction, conviction_from_pillars(pillars), confirming_reviews)
        return Recommendation(
            ticker=thesis.ticker, action="hold", size_change_pct=0.0, stop_price=thesis.stop_price,
            conviction_before=thesis.conviction, conviction_after=after,
            explanation="All three pillars are supportive; the thesis is intact.",
            evidence_ids=["fact:revenue_growth_yoy", "fact:px_technical"],
            falsifier="a missed checkpoint or a close below the 50-day average",
        )

    def _single_chain(self, thesis, facts, pillars, context, shock, last_review, confirming_reviews,
                      started, trace) -> ReviewResult:
        """Evaluation baseline: one linear chain of thought, no branches, no independent reviewer."""
        evidence = self.index.search("management explanation of results, risks, outlook and policy",
                                     ticker=thesis.ticker)
        self.researcher.calls += 1
        data = self.analyst.single_chain(context, evidence)
        branch = parse_branch({"id": "S", "explanation": data.get("explanation", ""),
                               "implied_action": data.get("implied_action", "hold"),
                               "pillar_effects": data.get("pillar_effects", {})}, depth=1)
        ids = [i for i in data.get("evidence_ids", []) if i in {e.chunk_id for e in evidence}]
        for e in evidence:
            e.label = "supports" if e.chunk_id in ids else "irrelevant"
        branch.evidence = evidence
        pillars_after = self._apply_effects(pillars, branch, trace)
        after = apply_upgrade_rule(thesis.conviction, conviction_from_pillars(pillars_after, shock), confirming_reviews)
        action = branch.implied_action
        size = {"hold": 0.0, "exit": -thesis.position_pct, "trim": -round(thesis.position_pct / 3, 2),
                "add": round(thesis.position_pct * 0.5, 2), "open": 0.0}[action]
        rec = Recommendation(ticker=thesis.ticker, action=action, size_change_pct=size,
                             stop_price=thesis.stop_price, conviction_before=thesis.conviction,
                             conviction_after=after, explanation=branch.explanation, evidence_ids=ids or
                             [e.chunk_id for e in evidence[:1]], falsifier=str(data.get("falsifier") or "not stated"))
        retrieved = {e.chunk_id: e for e in evidence}
        report = verify_recommendation(rec, facts, retrieved,
                                       extra_numbers=_thesis_numbers(thesis, thesis.stop_price, size))
        self._route(rec, thesis, report, False, 1.0, shock, sum(e.flagged_injection for e in evidence))
        return self._finish(rec, [branch], pillars_after, report, "single", started, trace)

    def _finish(self, rec, tree, pillars, report, mode, started, trace) -> ReviewResult:
        result = ReviewResult(
            recommendation=rec, branches=tree, pillars=pillars, guardrails=report, mode=mode,
            llm_calls=self.llm.n_calls - self._llm0, retrieval_calls=self.researcher.calls - self._ret0,
            seconds=round(time.time() - started, 2), incomplete=self.incomplete, trace=trace,
            retrieval_logs=[vars(log) for log in self.index.logs[self._log0:]],
        )
        if self.journal is not None:
            self.journal.save_review(result)
        return result


def _thesis_numbers(thesis: Thesis, stop: float, size_change: float) -> List[float]:
    """Numbers the owner's thesis and the plan itself supply; both modes may quote them."""
    cps = [float(v) for v in thesis.checkpoints.values() if isinstance(v, (int, float))]
    return [stop, abs(size_change), thesis.position_pct, thesis.stop_price] + cps + [v * 100 for v in cps]
