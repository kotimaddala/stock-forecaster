"""Every evaluation scenario must meet its expectations and never use isolated-out evidence."""

import pytest

from eval.harness import check, isolation_violations, run
from eval.scenario_data import all_scenarios

SCENARIOS = all_scenarios()


@pytest.mark.parametrize("scn", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario(scn):
    result = run(scn)
    assert check(scn, result) == []
    assert isolation_violations(scn, result) == []
    assert result.llm_calls <= 14 and result.retrieval_calls <= 12  # budgets hold


def test_demo_tree_shape():
    result = run(SCENARIOS[0])
    status = {b.branch_id: b.status for b in result.branches}
    assert status["C"] == "pruned" and status["A2"] == "pruned"   # contradicted by Tier 1 facts
    assert status["B1"] == "selected"
    rec = result.recommendation
    assert rec.stop_price > 41.0                                   # stop tightened
    assert rec.route == "human" and rec.evidence_ids


def test_single_chain_baseline_runs_on_same_inputs():
    result = run(SCENARIOS[0], mode="single")
    assert result.mode == "single" and result.llm_calls == 1
