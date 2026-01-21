Task: Implement D-lite historical baseline fetch for Top-10 symbols only (network calls minimized).

Repo: pnl-tracker
File: services/api/daily_close.py
Tests: packages/pnl/tests/test_risk_trend.py

GUARDRAILS:
- Sprint A is frozen. Do NOT modify Total/Day P&L logic or outputs.
- Do NOT change existing report sections or order.
- No snapshots or filesystem writes.
- Reuse existing price/history fetch utilities already present (do NOT add new dependencies).
- Minimal diff only.

Implementation:
- Determine Top-10 symbols by today's market value (reuse Sprint C computed values if available; otherwise compute from existing positions+prices using the same FX logic as Sprint C).
- Fetch historical close for those Top-10 symbols for target_date = today - 7 calendar days:
  - Query a small window (10–14 days) to find nearest prior trading day close.
  - If only 1 row exists (today only), treat baseline as missing for that symbol.
- Compute baseline market values using baseline closes and the SAME FX-to-TWD logic as today (document this in a comment).
- Compute baseline concentration weights (Top-10 subset), then baseline metrics: Top-1 weight and Top-3 sum.
- If fewer than 3 symbols have baseline closes, baseline is considered unavailable.

Tests:
- Mock the historical price fetch so tests do NOT hit network.
- Add tests:
  - baseline found when mock returns multiple rows
  - baseline missing when mock returns only today's row
- pytest -q must pass.

No Slack output changes in this step.

Commit message:
"Add D-lite: fetch 7D baseline closes for Top-10 symbols (no output change)"

Before coding: list exact functions/areas you will touch.