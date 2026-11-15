"""The one-screen reviewer card shown when a case is routed to the human (Checkpoint 6)."""

from __future__ import annotations

from .controller import ReviewResult


def render_card(result: ReviewResult) -> str:
    r = result.recommendation
    ev = {e.chunk_id: e for b in result.branches for e in b.evidence}
    lines = [
        f"## {r.ticker}: {r.action.upper()}"
        + (f" ({r.size_change_pct:+.2f}% of account)" if r.size_change_pct else ""),
        "",
        f"**Route:** {r.route} ({'; '.join(r.route_reasons)})",
        f"**Conviction:** {r.conviction_before} -> {r.conviction_after}   **Stop:** {r.stop_price:.2f}",
        "",
        r.explanation,
        "",
        "**Top evidence**",
    ]
    for cid in r.evidence_ids[:3]:
        e = ev.get(cid)
        if e is not None:
            snippet = " ".join(e.text.split()[:40])
            lines.append(f"- [{e.section}, filed {e.filed}, tier {e.tier}]({e.source_url}): {snippet}...")
        else:
            lines.append(f"- {cid}")
    if r.contested and r.runner_up:
        lines += ["", f"**Competing explanation (still close):** {r.runner_up}"]
    lines += [
        "",
        f"**What would change this view:** {r.falsifier or 'n/a'}",
        "**If you do nothing:** the existing position and its current stop are kept; no new position is opened.",
    ]
    if r.notes:
        lines += ["", "**Notes:** " + "; ".join(r.notes)]
    lines += ["", "**Choose:** Approve | Approve with changes (size or stop) | Skip",
              "Reason: agree | too_extended | distrust_catalyst | size_too_large | timing | other"]
    return "\n".join(lines)
