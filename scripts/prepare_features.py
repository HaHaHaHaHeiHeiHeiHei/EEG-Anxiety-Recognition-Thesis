"""中文说明

用途：
    给出公开数据集特征提取入口的统一帮助信息。
输入：
    无；实际提取请运行对应 `python -m anxiety_eeg.features.score_*` 模块。
输出：
    控制台打印命令清单。
快速运行：
    `python scripts/prepare_features.py`
论文对应：
    第 3、4 章数据与特征构建。
注意事项：
    公开仓库不包含真实数据，需先按 `data/<dataset>/README.md` 下载。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401


def main() -> int:
    commands = [
        "python -m anxiety_eeg.features.score_ds003478 --ds-root data/ds003478 --output-dir features/subject_features/ds003478",
        "python -m anxiety_eeg.features.score_ds007609 --output-dir features/subject_features/ds007609",
        "python -m anxiety_eeg.features.score_ds007216 --output-dir features/external/ds007216",
        "python -m anxiety_eeg.external.extract_mendeley_subject_features --workbook data/mendeley_anxiety_control/EEG_data.xlsx",
    ]
    print("特征提取命令示例：")
    for command in commands:
        print(f"  {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
