"""中文说明

用途：
    薄入口：训练论文主模型 JointConstraintNet。
输入：
    见 `python scripts/train_joint.py --help`。
输出：
    默认写入 `outputs/joint_constraint`。
快速运行：
    `python scripts/train_joint.py --features-root tests/fixtures/subject_features --seeds 42 --epochs 1 --min-epochs 1 --patience 1 --device cpu`
论文对应：
    第 4、5 章。
注意事项：
    真实复现请把 `--features-root` 指向完整受试者级特征表目录。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from anxiety_eeg.training.train_joint import main


if __name__ == "__main__":
    raise SystemExit(main())
