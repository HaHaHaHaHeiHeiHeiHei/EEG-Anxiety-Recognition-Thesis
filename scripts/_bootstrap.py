"""中文说明

用途：
    让 `python scripts/*.py` 在未安装 editable 包时也能找到 `src/anxiety_eeg`。
输入：
    无。
输出：
    将仓库 `src` 目录加入 `sys.path`。
快速运行：
    被其他脚本自动导入，不需要单独运行。
论文对应：
    工程复现辅助，不对应具体论文章节。
注意事项：
    安装 `pip install -e .` 后该文件仍然安全可用。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
