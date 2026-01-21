# Prompts（給 Claude Code 的任務腳本）

這個目錄放「可重複使用」的 Claude 任務提示句，用來確保：
- 每次改動最小化（Minimal diff）
- 不破壞已封存的 Sprint A
- append-only 的輸出規則一致
- 測試可離線（mock network）

## 使用方式（標準流程）
1. 開分支（可選，但建議）
2. 打開 Claude Code
3. 複製貼上對應的 prompt（依序做）
4. 跑測試：`pytest -q`
5. 檢查改動檔案：`git diff --name-only`
6. commit + push

## D-lite（Top-10）流程
1) `D-lite-1-helpers-and-tests.md`
2) `D-lite-2-fetch-baseline-top10.md`
3) `D-lite-3-append-slack-trend.md`