"""Long-term memory (Checkpoint 2.1): an SQLite journal.

Stores every review with its full reasoning tree, every human decision with a
reason code, shadow positions for missed-opportunity tracking, and lessons
that move through candidate -> validated -> retired. Only validated lessons
are ever returned to agents.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY, created TEXT, ticker TEXT, mode TEXT, action TEXT,
  conviction_before TEXT, conviction_after TEXT, route TEXT, fallback INTEGER,
  llm_calls INTEGER, retrieval_calls INTEGER, payload TEXT);
CREATE TABLE IF NOT EXISTS nodes (
  review_id INTEGER, branch_id TEXT, parent_id TEXT, depth INTEGER, explanation TEXT,
  implied_action TEXT, score REAL, status TEXT, prune_reason TEXT, evidence TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, review_id INTEGER, created TEXT, decision TEXT, reason_code TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS shadow_positions (
  id INTEGER PRIMARY KEY, review_id INTEGER, ticker TEXT, opened TEXT, price REAL, why TEXT, closed TEXT, exit_price REAL);
CREATE TABLE IF NOT EXISTS lessons (
  id INTEGER PRIMARY KEY, created TEXT, tag TEXT, text TEXT, stage TEXT,
  cases INTEGER, success_rate REAL, market_adjusted INTEGER, out_of_sample INTEGER, reason TEXT);
"""

REASON_CODES = ("agree", "too_extended", "distrust_catalyst", "size_too_large", "timing", "other")
MIN_CASES = 8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path: Path | str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def save_review(self, result) -> int:
        rec = result.recommendation
        payload = {"recommendation": rec.to_dict(), "pillars": result.pillars, "trace": result.trace,
                   "violations": result.guardrails.violations, "retrieval_logs": result.retrieval_logs}
        cur = self.db.execute(
            "INSERT INTO reviews (created, ticker, mode, action, conviction_before, conviction_after, route, "
            "fallback, llm_calls, retrieval_calls, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), rec.ticker, result.mode, rec.action, rec.conviction_before, rec.conviction_after,
             rec.route, int(rec.fallback), result.llm_calls, result.retrieval_calls, json.dumps(payload, default=str)))
        rid = cur.lastrowid
        for b in result.branches:
            self.db.execute(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                (rid, b.branch_id, b.parent_id, b.depth, b.explanation, b.implied_action, b.score, b.status,
                 b.prune_reason, json.dumps([{"id": e.chunk_id, "label": e.label, "tier": e.tier,
                                               "filed": e.filed} for e in b.evidence])))
        self.db.commit()
        return int(rid)

    def record_decision(self, review_id: int, decision: str, reason_code: str, note: str = "") -> None:
        if decision not in ("approve", "approve_with_changes", "skip"):
            raise ValueError("decision must be approve, approve_with_changes or skip")
        if reason_code not in REASON_CODES:
            raise ValueError(f"reason_code must be one of {REASON_CODES}")
        self.db.execute("INSERT INTO decisions (review_id, created, decision, reason_code, note) VALUES (?,?,?,?,?)",
                        (review_id, _now(), decision, reason_code, note))
        self.db.commit()

    def open_shadow(self, review_id: int, ticker: str, price: float, why: str) -> None:
        self.db.execute("INSERT INTO shadow_positions (review_id, ticker, opened, price, why) VALUES (?,?,?,?,?)",
                        (review_id, ticker, _now()[:10], price, why))
        self.db.commit()

    # ---- lessons ------------------------------------------------------------
    def propose_lesson(self, tag: str, text: str, cases: int, success_rate: float,
                       market_adjusted: bool, out_of_sample: bool, reason: str) -> str:
        """A lesson is validated only if it passes all four tests; otherwise it stays a candidate."""
        stage = "validated" if (cases >= MIN_CASES and market_adjusted and out_of_sample and reason.strip()) \
            else "candidate"
        self.db.execute("INSERT INTO lessons (created, tag, text, stage, cases, success_rate, market_adjusted, "
                        "out_of_sample, reason) VALUES (?,?,?,?,?,?,?,?,?)",
                        (_now(), tag, text, stage, cases, success_rate, int(market_adjusted), int(out_of_sample), reason))
        self.db.commit()
        return stage

    def lessons_for(self, tag: str, limit: int = 3) -> List[str]:
        rows = self.db.execute("SELECT text FROM lessons WHERE stage='validated' AND tag=? "
                               "ORDER BY cases DESC LIMIT ?", (tag, limit)).fetchall()
        return [r[0] for r in rows]

    def retire_lesson(self, lesson_id: int) -> None:
        self.db.execute("UPDATE lessons SET stage='retired' WHERE id=?", (lesson_id,))
        self.db.commit()

    # ---- monitoring -----------------------------------------------------------
    def summary(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchall()  # noqa: E731
        routes = dict(q("SELECT route, COUNT(*) FROM reviews GROUP BY route"))
        total = sum(routes.values()) or 1
        decisions = dict(q("SELECT decision, COUNT(*) FROM decisions GROUP BY decision"))
        return {
            "reviews": sum(routes.values()), "routes": routes,
            "escalation_rate": round(routes.get("human", 0) / total, 3),
            "fallbacks": q("SELECT COUNT(*) FROM reviews WHERE fallback=1")[0][0],
            "decisions": decisions,
            "approval_rate": round(decisions.get("approve", 0) / max(1, sum(decisions.values())), 3),
            "lessons": dict(q("SELECT stage, COUNT(*) FROM lessons GROUP BY stage")),
        }
