"""中文说明

用途：
    使用合成 fixture 跑最小主模型和 LogReg-L2 基线，验证仓库代码可执行。
输入：
    `tests/fixtures/subject_features` 中的合成 `subject_features.csv`。
输出：
    默认写入 `outputs/smoke/joint` 和 `outputs/smoke/baselines`。
快速运行：
    `python scripts/run_smoke.py --device cpu`
论文对应：
    工程验证，不对应论文实验结果。
注意事项：
    smoke 数据是人工合成的，只能证明代码链路可跑，不能作为论文指标。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "subject_features"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "smoke"


def run(command: list[str]) -> None:
    print("\n[RUN] " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal smoke tests for the thesis code repository.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    joint_out = args.output_root / "joint"
    baseline_out = args.output_root / "baselines"
    run(
        [
            sys.executable,
            "scripts/train_joint.py",
            "--features-root",
            str(args.features_root),
            "--output-root",
            str(joint_out),
            "--seeds",
            "42",
            "--epochs",
            "1",
            "--min-epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "8",
            "--device",
            args.device,
            "--experiment-name",
            "smoke_joint",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/train_baselines.py",
            "--features-root",
            str(args.features_root),
            "--output-root",
            str(baseline_out),
            "--seeds",
            "42",
            "--models",
            "logreg_l2",
            "--n-jobs",
            "1",
        ]
    )
    print(f"\nSmoke finished. Outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
