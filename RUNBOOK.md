	•	如何手動觸發 Daily Close
	•	常見錯誤：Google 權限、匯率 API、yfinance 查不到台股
	•	失敗時的處置：重跑、跳過、或 fail fast


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