#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文说明

用途：
    从 OpenNeuro ds003478 原始 EEG 中提取论文主模型需要的受试者级频谱特征。
输入：
    `--ds-root` 指向下载后的 ds003478 数据目录。
输出：
    默认写入 `features/subject_features/ds003478/subject_features.csv` 及审计文件。
快速运行：
    `python -m anxiety_eeg.features.score_ds003478 --ds-root data/ds003478`
论文对应：
    第 3 章数据构建和第 4 章全局频谱组织。
注意事项：
    原始数据不随仓库分发，需要先按 `data/ds003478/README.md` 下载。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anxiety_eeg.features.common_dataset_eval import (
    DEFAULT_DS_ROOT,
    add_common_cli_arguments,
    collect_ds003478_bundles,
    default_scoring_dir,
    run_dataset_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ds003478 anxiety-related EEG global features.")
    parser.add_argument("--ds-root", type=Path, default=DEFAULT_DS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=default_scoring_dir("ds003478"))
    parser.add_argument("--ds-run", choices=["01", "02"], default="01")
    parser.add_argument("--ds-condition", choices=["open", "closed", "all"], default="all")
    parser.add_argument("--limit-subjects", type=int, default=None)
    add_common_cli_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = {
        "ds_run": args.ds_run,
        "ds_condition": args.ds_condition,
        "window_sec": float(args.window_sec),
        "max_windows": int(args.max_windows),
        "nperseg_sec": float(args.nperseg_sec),
        "use_common_local_channels": bool(args.use_common_local_channels),
        "limit_subjects": args.limit_subjects,
        "min_n": int(args.min_n),
    }
    return run_dataset_pipeline(
        dataset_name="ds003478",
        score_name="STAI",
        source_root=args.ds_root,
        output_dir=args.output_dir.resolve(),
        min_n=args.min_n,
        config=config,
        collector=lambda: collect_ds003478_bundles(
            ds_root=args.ds_root,
            ds_run=args.ds_run,
            ds_condition=args.ds_condition,
            window_sec=args.window_sec,
            max_windows=args.max_windows,
            nperseg_sec=args.nperseg_sec,
            use_common_local_channels=args.use_common_local_channels,
            limit_subjects=args.limit_subjects,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
