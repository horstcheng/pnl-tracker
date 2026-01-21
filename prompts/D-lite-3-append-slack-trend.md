Task: Append "風險趨勢（7日｜Top-10）" section to Slack report (D-lite), using Top-10 baseline metrics.

Repo: pnl-tracker
File: services/api/daily_close.py
Tests: packages/pnl/tests/test_risk_trend.py

GUARDRAILS (must follow):
- Sprint A is frozen. Do NOT modify Total/Day P&L logic, formulas, ordering, or existing outputs.
- Do NOT change existing report sections or their order. Only APPEND at the very end (after Risk Views).
- No snapshots. No filesystem writes.
- No new dependencies. Minimal diff only.

Output format (append-only, zh-TW):
At the very end append:

"風險趨勢（7日｜Top-10）："
  "Top-1 集中度：{base:.1f}% → {today:.1f}%（{delta:+.1f}%）"
  "Top-3 集中度：{base:.1f}% → {today:.1f}%（{delta:+.1f}%）"

If baseline unavailable:
"風險趨勢（7日｜Top-10）：無法取得足夠的歷史價格，略過。"

Notes:
- Today metrics must come from today’s concentration (Sprint C results).
- Baseline metrics from D-lite Top-10 baseline.
- Percentages with 1 decimal place.

Tests:
- Unit test formatting and delta signs given known baseline/today metrics.
- pytest -q must pass.

Commit message:
"Add Sprint D-lite: concentration risk trend (7D Top-10, append-only)"

Before coding: describe exactly which lines/areas you will touch.