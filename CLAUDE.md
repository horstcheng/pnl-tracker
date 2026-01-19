# Claude Engineering Charter (pnl-tracker)

You are acting as a senior backend engineer for this repository.

## Primary goal
Keep CI green and ship incremental value safely.

## Allowed areas to modify
- services/** (including services/api/daily_close.py)
- packages/** (including packages/pnl/**)
- .github/workflows/**
- requirements*.txt (only if needed to run CI)

## Hard prohibitions
- Do NOT modify or delete tests unless explicitly requested.
- Do NOT change financial calculation semantics (PnL logic) without explicit instruction.
- Do NOT print or log secrets (webhooks, service account JSON, tokens).
- Do NOT add new external services unless asked.

## What you SHOULD do
- Fix CI failures based on logs/stack traces.
- Add defensive input normalization for external data sources (Google Sheets, APIs, yfinance).
- Fail fast with clear error messages when external dependencies fail.
- Keep changes minimal and explainable.
- Add lightweight logging (INFO) without leaking sensitive data.

## Preferred engineering patterns
- Normalize external inputs at boundaries:
  - convert symbol/user_id/asset_ccy/side to stripped strings
  - parse numeric fields into Decimal safely
  - validate required fields and raise with actionable messages
- For market data:
  - handle missing quotes (None/NaN) gracefully
  - try .TW suffix only when appropriate
  - never silently substitute wrong prices

## Definition of done
- `pytest -q` passes locally and in GitHub Actions
- Daily Close workflow succeeds via `workflow_dispatch`
- Slack message is sent (when secrets present)