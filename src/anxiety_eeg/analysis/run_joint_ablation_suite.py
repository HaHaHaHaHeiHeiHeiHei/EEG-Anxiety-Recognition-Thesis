"""中文说明

用途：
    批量运行论文中的结构消融：full_main、shared_subspace、no_adapter、
    no_global_aux、no_frontal_aux、no_shared_constraints、no_gray_weighting、
    global_threshold。
输入：
    `--features-root` 指向真实受试者级特征目录；可选已训练 checkpoint 后再跑外部验证。
输出：
    每个消融实验的训练输出和可选外部验证结果，默认写入 `outputs/joint_ablation_suite`。
快速运行：
    `python scripts/run_ablations.py --features-root features/subject_features --device cpu --seeds 42 --skip-external`
论文对应：
    第 5 章结构消融与 gray-zone 敏感性分析。
注意事项：
    完整多 seed 消融耗时较长；smoke 测试不调用本脚本。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES_ROOT = REPO_ROOT / "features" / "subject_features"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "joint_ablation_suite"
DEFAULT_SEEDS = [42, 123, 224, 3407, 65422, 7, 21, 2024, 31415, 27182]


EXPERIMENTS: dict[str, dict[str, object]] = {
    "full_main": {
        "description": "Main cross-dataset model with full features and both auxiliary constraints.",
        "train_args": [
            "--experiment-name",
            "full_main",
            "--feature-preset",
            "full",
        ],
    },
    "shared_subspace": {
        "description": "Shared common theta/alpha/beta feature subspace used across all datasets.",
        "train_args": [
            "--experiment-name",
            "shared_subspace",
            "--feature-preset",
            "shared_common",
        ],
    },
    "no_adapter": {
        "description": "Shared trunk only, with adapter and dataset bias zeroed and frozen.",
        "train_args": [
            "--experiment-name",
            "no_adapter",
            "--freeze-adapter",
            "--freeze-dataset-bias",
        ],
    },
    "no_global_aux": {
        "description": "Disable the global beta-dominant auxiliary loss.",
        "train_args": [
            "--experiment-name",
            "no_global_aux",
            "--global-loss-weight",
            "0.0",
        ],
    },
    "no_frontal_aux": {
        "description": "Disable the frontal beta auxiliary loss.",
        "train_args": [
            "--experiment-name",
            "no_frontal_aux",
            "--region-loss-weight",
            "0.0",
        ],
    },
    "no_shared_constraints": {
        "description": "Remove both auxiliary constraints while keeping the classifier route.",
        "train_args": [
            "--experiment-name",
            "no_shared_constraints",
            "--global-loss-weight",
            "0.0",
            "--region-loss-weight",
            "0.0",
        ],
    },
    "no_gray_weighting": {
        "description": "Keep gray-zone samples at full weight instead of reduced weight.",
        "train_args": [
            "--experiment-name",
            "no_gray_weighting",
            "--gray-zone-weight",
            "1.0",
        ],
    },
    "global_threshold": {
        "description": "Use fixed per-dataset global medians rather than train-only medians.",
        "train_args": [
            "--experiment-name",
            "global_threshold",
            "--threshold-mode",
            "dataset_global_median",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cleanly named joint ablation suite for the paper.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()))
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(command))
    env = os.environ.copy()
    src_dir = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    unknown = [name for name in args.experiments if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")

    for exp_name in args.experiments:
        spec = EXPERIMENTS[exp_name]
        exp_dir = output_root / exp_name
        train_root = exp_dir / "train_outputs"
        mendeley_root = exp_dir / "external_mendeley"
        ds007216_root = exp_dir / "external_ds007216"
        print(f"\n========== {exp_name} ==========")
        print(spec["description"])

        if not args.skip_training:
            train_cmd = [
                str(args.python),
                "-m",
                "anxiety_eeg.training.train_joint",
                "--features-root",
                str(args.features_root),
                "--output-root",
                str(train_root),
                "--device",
                str(args.device),
                "--seeds",
                *[str(seed) for seed in args.seeds],
                *[str(item) for item in spec["train_args"]],
            ]
            run_command(train_cmd, cwd=REPO_ROOT)

        if args.skip_external:
            continue

        mendeley_cmd = [
            str(args.python),
            "-m",
            "anxiety_eeg.external.evaluate_external_mendeley",
            "--train-output-root",
            str(train_root),
            "--results-dir",
            str(mendeley_root),
            "--skip-scoring",
            "--device",
            str(args.device),
        ]
        run_command(mendeley_cmd, cwd=REPO_ROOT)

        ds007216_cmd = [
            str(args.python),
            "-m",
            "anxiety_eeg.external.evaluate_external_ds007216",
            "--train-output-root",
            str(train_root),
            "--results-dir",
            str(ds007216_root),
            "--skip-scoring",
            "--device",
            str(args.device),
        ]
        run_command(ds007216_cmd, cwd=REPO_ROOT)

    print(f"\nFinished. Outputs saved under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
