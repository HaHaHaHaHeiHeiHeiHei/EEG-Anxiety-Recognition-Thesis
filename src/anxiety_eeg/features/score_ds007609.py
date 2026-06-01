#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文说明

用途：
    通过 OpenNeuro/EEGDash 提取 ds007609 的 trait-anxiety EEG 频谱特征。
输入：
    公开数据集 ds007609，可通过脚本自动缓存到 `data/_cache/eegdash`。
输出：
    默认写入 `features/subject_features/ds007609/subject_features.csv` 及审计文件。
快速运行：
    `python -m anxiety_eeg.features.score_ds007609 --output-dir features/subject_features/ds007609`
论文对应：
    第 3 章数据构建和第 4 章全局频谱组织。
注意事项：
    首次运行可能下载较多数据，请确认网络、磁盘和数据许可。
"""

from __future__ import annotations

import argparse

from anxiety_eeg.features.eegdash_remote_eval import EEGDashDatasetSpec, add_remote_cli_arguments, run_remote_dataset_pipeline


SPEC = EEGDashDatasetSpec(
    dataset_name="ds007609",
    eegdash_dataset_id="ds007609",
    openneuro_dataset_id="ds007609",
    score_name_hint="trait_anxiety",
    preferred_label_columns=(
        "stai_trait",
        "trait_anxiety",
        "STAI",
        "stai",
        "stai_state",
        "traitanxiety",
        "anxiety",
    ),
    task_include_keywords=(),
    task_exclude_keywords=(),
    target_sfreq=250.0,
    notes="EEGDash/OpenNeuro resting-state trait-anxiety dataset.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEGDash ds007609 trait-anxiety feasibility analysis.")
    add_remote_cli_arguments(parser, SPEC)
    return parser.parse_args()


def main() -> int:
    return run_remote_dataset_pipeline(parse_args(), SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
