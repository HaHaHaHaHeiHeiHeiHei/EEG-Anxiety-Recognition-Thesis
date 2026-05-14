"""中文说明

用途：
    提供轻量配置读取工具，让训练脚本支持 `--config` JSON 文件。
输入：
    JSON 配置文件，键名与 argparse 参数名一致，例如 `features_root`、`epochs`。
输出：
    更新后的 argparse Namespace。
快速运行：
    `python scripts/train_joint.py --config configs/smoke.json`
论文对应：
    复现实验配置管理，不直接对应某一节。
注意事项：
    为了减少额外依赖，这里只解析 JSON；命令行参数和配置同时给出时，
    配置文件会覆盖同名参数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "features_root",
    "output_root",
    "train_output_root",
    "scoring_dir",
    "results_dir",
    "workbook",
    "label_file",
    "data_cache_dir",
}


def apply_json_config(args: argparse.Namespace) -> argparse.Namespace:
    """Apply `args.config` JSON values to an argparse namespace."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        return args
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    for key, value in data.items():
        if not hasattr(args, key):
            continue
        if key in PATH_KEYS and value is not None:
            value = Path(value)
        setattr(args, key, value)
    return args
