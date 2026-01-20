	•	如何手動觸發 Daily Close
	•	常見錯誤：Google 權限、匯率 API、yfinance 查不到台股
	•	失敗時的處置：重跑、跳過、或 fail fast


## P&L Calculation

### Total P&L

**Definition:** The cumulative profit/loss for each position from acquisition to current close.

**Formula:**
```
Total P&L = Realized P&L + Unrealized P&L

Realized P&L = sum of (sell_price - avg_cost) * sell_quantity - fees (for closed positions)
Unrealized P&L = (current_close - avg_cost) * remaining_quantity (for open positions)

where avg_cost = weighted average cost including fees
```

**Data Sources:**
- Trade history: Google Sheets (`trades` tab)
- Current close price: yfinance (`ticker.history(period="7d")`, last valid Close)
- Exchange rate: exchangerate-api.com (`/v4/latest/USD`)

**Currency Conversion:**
- TWD positions: P&L reported in TWD (no conversion)
- USD positions: P&L converted to TWD using daily USD/TWD rate

---

### Day P&L

**Definition:** The daily change in portfolio value based on price movement from previous close to today's close.

**Formula:**
```
Day P&L = (today_close - prev_close) * current_position_quantity
```

**Data Sources (in order of priority):**

1. **Primary:** yfinance daily history
   - Fetch 7 days of history: `ticker.history(period="7d")`
   - Use last two valid Close values as `(prev_close, today_close)`

2. **Fallback:** yfinance fast_info (when history has < 2 rows)
   - `prev_close = ticker.fast_info.get("previousClose")`
   - `today_close` from history (last valid Close)

3. **Lenient mode:** (when both primary and fallback fail)
   - `day_pnl = 0` for that symbol
   - Symbol added to `missing_prev_close` warnings
   - Report includes note: "missing prev_close symbols are treated as 0"

**Currency Conversion:**
- Same as Total P&L (TWD positions in TWD, USD positions converted via USD/TWD rate)

---

### Known Limitations

1. **yfinance history may return only one row for some TW ETFs**
   - Observed for: 00983A.TW, 00984A.TW, 00988A.TW
   - Root cause: yfinance data availability for certain Taiwan-listed ETFs
   - Mitigation: fallback to `fast_info.previousClose`

2. **Fallback may also fail**
   - If `fast_info.previousClose` is unavailable, Day P&L defaults to 0
   - Symbol appears in warnings with debug info (ticker, rows, dates, closes)

3. **Market holidays and data gaps**
   - 7-day history window accommodates most holidays
   - If fewer than 2 trading days in window, fallback is triggered


====BUG 修復指令====
CI is failing. Please fix the bug.

Rules:
- Follow CLAUDE.md
- Keep changes minimal
- Do not modify tests
- Do not leak secrets in logs
- After the fix, run pytest -q and ensure it passes
- Commit with a clear message and open a PR
Here is the error log:
<paste the red error block>

流程：
	1.	CI 紅燈
	2.	你把 log 貼到 Claude Code（用上面模板）
	3.	Claude 直接在 Codespaces 修、跑測試、push 分支、開 PR
	4.	你在 iPad 上審一下 → merge


   ## Sprint C – Risk Views (Completed)

**Status:** ✅ Completed  
**Release:** v0.3.0  
**Date:** 2026-01-19

### Scope
- Added portfolio risk views to the daily Slack report (append-only).
- No changes to existing P&L logic or outputs.

### Features
1. **Concentration Risk (by market value)**
   - Calculates per-symbol market value using:
     `quantity × today_close × fx_to_twd`
   - Displays Top 5 symbols by portfolio weight.
   - Includes alerts:
     - ⚠️ Top-1 concentration risk if weight > 25%
     - ⚠️ Top-3 concentration risk if combined weight > 60%

2. **Currency Exposure**
   - Aggregates portfolio market value by asset currency.
   - Displays percentage and total TWD value per currency.

### Guardrails (Enforced)
- Sprint A (Total / Day P&L) logic frozen and unchanged.
- No new price or FX fetches introduced.
- Existing report sections and order preserved.
- Risk Views appended at the end of the report only.

### Files Changed
- `services/api/daily_close.py`
- `packages/pnl/tests/test_risk_views.py`

### Validation
- `pytest -q` passes.n

- Slack report verified with live data.