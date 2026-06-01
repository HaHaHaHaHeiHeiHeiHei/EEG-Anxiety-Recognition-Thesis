"""中文说明

用途：
    给出论文最终数据协议所需特征表的统一准备清单。
输入：
    无；OpenNeuro 原始 EEG 到特征表的提取器未随当前仓库分发。
输出：
    控制台打印内部特征表目标路径和可运行的 Mendeley 提取命令。
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
    internal_feature_paths = [
        "features/subject_features/original_local/subject_features.csv  # EVA-MED",
        "features/subject_features/ds003478/subject_features.csv",
        "features/subject_features/ds007609/subject_features.csv",
    ]
    print("内部训练/验证特征表目标路径：")
    for path in internal_feature_paths:
        print(f"  {path}")
    print("\n说明：OpenNeuro 原始 EEG 到 subject_features.csv 的提取器未随当前仓库分发。")
    print("\nMendeley 外部验证特征提取命令：")
    print(
        "  python -m anxiety_eeg.external.extract_mendeley_subject_features "
        "--workbook data/mendeley_anxiety_control/EEG_data.xlsx"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
