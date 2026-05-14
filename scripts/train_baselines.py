"""中文说明

用途：
    薄入口：训练传统机器学习基线。
输入：
    见 `python scripts/train_baselines.py --help`。
输出：
    默认写入 `outputs/traditional_baselines`。
快速运行：
    `python scripts/train_baselines.py --features-root tests/fixtures/subject_features --seeds 42 --models logreg_l2 --n-jobs 1`
论文对应：
    第 5 章主模型与传统基线比较。
注意事项：
    与主模型共享 subject-level split 和 gray-zone 标签规则。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from anxiety_eeg.training.train_baselines import main


if __name__ == "__main__":
    raise SystemExit(main())
