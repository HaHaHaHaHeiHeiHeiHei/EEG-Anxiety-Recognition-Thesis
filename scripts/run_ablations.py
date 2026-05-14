"""中文说明

用途：
    薄入口：运行结构消融和 gray-zone 敏感性实验。
输入：
    见 `python scripts/run_ablations.py --help`。
输出：
    默认写入 `outputs/joint_ablation_suite`。
快速运行：
    `python scripts/run_ablations.py --features-root features/subject_features --seeds 42 --skip-external --device cpu`
论文对应：
    第 5 章结构消融。
注意事项：
    完整消融耗时较长，建议先用单 seed 检查环境。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from anxiety_eeg.analysis.run_joint_ablation_suite import main


if __name__ == "__main__":
    raise SystemExit(main())
