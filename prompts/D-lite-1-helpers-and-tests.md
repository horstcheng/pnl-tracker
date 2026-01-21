Task: Add pure helper functions to compute concentration weights and 7D trend deltas (NO network, NO report changes yet).

Repo: pnl-tracker
File: services/api/daily_close.py
Tests: packages/pnl/tests/test_risk_trend.py (create if missing)

GUARDRAILS:
- Sprint A is frozen. Do NOT modify Total/Day P&L logic or outputs.
- Do NOT change existing report sections or order.
- No network calls in this step.
- Minimal diff only.

Implement (pure functions only):
1) compute_market_values_twd(positions, prices, usd_twd) -> dict[symbol] = value_twd
   - Reuse the same currency/FX-to-TWD logic already used in Sprint C. Do NOT change it.
2) compute_concentration_from_values(values_twd) -> list[(symbol, pct, value_twd)] sorted by pct desc
3) compute_topk_concentration_metrics(concentration, k=3) -> (top1_pct, top3_pct_sum)
4) compute_trend_pp(base_pct, today_pct) -> delta_pp

Tests (no network):
- concentration weights: sum ≈ 100%
- sorting descending
- top1/top3 metrics correct
- trend delta (+/-) correct

No Slack output changes in this step.
pytest -q must pass.

Commit message:
"Add risk trend helpers (pure functions, no behavior change)"

Before coding: list exact line ranges you will touch.