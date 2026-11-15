---
name: thesis-review
description: Review a saved stock thesis when new filings, earnings or policy news may have changed it. Use when the user asks whether to hold, trim or exit a position they track.
---

# Thesis Review

## When to use
- A tracked company filed a 10-Q, 10-K or 8-K, or a related policy changed.
- The user asks "should I still hold X?" for a thesis saved under `examples/` or their own folder.

## Steps
1. Call `review_thesis` with the thesis file path and a one-line trigger.
2. Show the reviewer card as returned. Do not restate numbers that are not in it.
3. If the route is `human`, ask the user to choose Approve, Approve with changes, or Skip, and record it
   with `python -m forecaster.cli decide <review_id> <decision> <reason>`.
4. Never present the result as financial advice, and never suggest placing an order outside the approval flow.

## Notes
- The tool is read-only. There is no trading tool by design.
- "Insufficient evidence; conviction unchanged" is a valid, safe outcome, not an error.
