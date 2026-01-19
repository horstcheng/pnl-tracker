# conftest.py (repo root)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
root_str = str(ROOT)

# 確保 repo 根目錄永遠在 sys.path 最前面
if root_str not in sys.path:
    sys.path.insert(0, root_str)