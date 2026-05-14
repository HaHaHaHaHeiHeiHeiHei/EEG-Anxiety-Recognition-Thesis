"""中文说明

用途：
    训练 shared_subspace 的 Logistic Regression 参考基线，用来判断跨数据集共有
    theta/alpha/beta 特征子空间中是否存在弱但可用的焦虑相关信号。
输入：
    `--features-root` 为内部三数据集特征目录；外部 CSV 可通过
    `--mendeley-features` 和 `--ds007216-features` 指定。
输出：
    内部验证、Mendeley 兼容性验证和 ds007216 方向审计的 CSV summary。
快速运行：
    `python -m anxiety_eeg.analysis.run_shared_subspace_logistic --features-root features/subject_features --skip-external`
论文对应：
    第 5 章探索性 LODO/共享子空间和外部兼容性分析。
注意事项：
    若没有外部特征表，请使用 `--skip-external`，内部结果仍可复现。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

from anxiety_eeg.config import apply_json_config
from anxiety_eeg.data.joint_dataset import (  # noqa: E402
    DEFAULT_FEATURES_ROOT,
    build_joint_datasets,
    dataset_csvs_from_root,
    read_subject_feature_csv,
)


DEFAULT_SEEDS = [42, 123, 224, 3407, 65422, 7, 21, 2024, 31415, 27182]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "shared_subspace_logistic"
DEFAULT_MENDELEY_CSV = REPO_ROOT / "features" / "external" / "mendeley" / "subject_features.csv"
DEFAULT_DS007216_CSV = REPO_ROOT / "features" / "external" / "ds007216" / "subject_features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a shared-subspace logistic baseline.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config file; keys match CLI argument names.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mendeley-features", type=Path, default=DEFAULT_MENDELEY_CSV)
    parser.add_argument("--ds007216-features", type=Path, default=DEFAULT_DS007216_CSV)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--feature-preset", choices=["shared_common"], default="shared_common")
    parser.add_argument("--threshold-mode", choices=["train_median", "dataset_global_median"], default="train_median")
    parser.add_argument("--gray-z", type=float, default=0.35)
    parser.add_argument("--gray-zone-weight", type=float, default=0.35)
    return apply_json_config(parser.parse_args())


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_metrics(labels: np.ndarray, probs: np.ndarray, in_gray_zone: np.ndarray | None = None) -> dict[str, float]:
    pred = (probs >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": safe_roc_auc(labels, probs),
        "pr_auc": safe_pr_auc(labels, probs),
    }
    if in_gray_zone is not None:
        mask = ~in_gray_zone
        if np.sum(mask) >= 3 and len(np.unique(labels[mask])) > 1:
            metrics["extreme_balanced_accuracy"] = float(balanced_accuracy_score(labels[mask], pred[mask]))
            metrics["extreme_roc_auc"] = safe_roc_auc(labels[mask], probs[mask])
        else:
            metrics["extreme_balanced_accuracy"] = float("nan")
            metrics["extreme_roc_auc"] = float("nan")
    return metrics


def selection_score(metrics: dict[str, float]) -> float:
    def finite(x: float | None) -> float:
        if x is None or not math.isfinite(float(x)):
            return 0.0
        return float(x)

    return (
        0.35 * finite(metrics.get("extreme_roc_auc"))
        + 0.25 * finite(metrics.get("extreme_balanced_accuracy"))
        + 0.25 * finite(metrics.get("roc_auc"))
        + 0.15 * finite(metrics.get("balanced_accuracy"))
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def pooled_normalizer(split_info: dict) -> tuple[dict[str, float], dict[str, float]]:
    feature_names = list(split_info["input_features"])
    per_dataset = dict(split_info["per_dataset"])
    counts = {name: int(info["train_count"]) for name, info in per_dataset.items()}
    total = float(sum(counts.values()))

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature_name in feature_names:
        weighted_mean = 0.0
        for dataset_name, info in per_dataset.items():
            weighted_mean += counts[dataset_name] * float(info["normalizer"]["feature_means"][feature_name])
        mean = weighted_mean / max(total, 1.0)
        weighted_second = 0.0
        for dataset_name, info in per_dataset.items():
            std = float(info["normalizer"]["feature_stds"][feature_name])
            mu = float(info["normalizer"]["feature_means"][feature_name])
            weighted_second += counts[dataset_name] * (std * std + mu * mu)
        var = max(weighted_second / max(total, 1.0) - mean * mean, 1e-8)
        means[feature_name] = float(mean)
        stds[feature_name] = float(np.sqrt(var))
    return means, stds


def build_external_matrix(rows: list, feature_names: list[str], means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append(
            [
                (float(row.features[name]) - float(means[name])) / (float(stds[name]) + 1e-8)
                for name in feature_names
            ]
        )
    return np.asarray(matrix, dtype=np.float32)


def compute_external_thresholds(rows: list, gray_z: float) -> dict[str, float]:
    scores = np.asarray([float(row.anxiety) for row in rows], dtype=np.float32)
    center = float(np.median(scores))
    std = float(np.std(scores))
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    return {
        "center": center,
        "std": std,
        "low_threshold": float(center - gray_z * std),
        "high_threshold": float(center + gray_z * std),
    }


def aggregate_mean_std(rows: list[dict], metric_names: list[str]) -> list[dict]:
    out = []
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        out.append(
            {
                "metric": metric,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    internal_rows: list[dict] = []
    internal_prediction_rows: list[dict] = []
    mendeley_summary_rows: list[dict] = []
    mendeley_prediction_rows: list[dict] = []
    ds007216_summary_rows: list[dict] = []
    ds007216_prediction_rows: list[dict] = []

    dataset_csvs = dataset_csvs_from_root(args.features_root)
    mendeley_rows = []
    ds007216_rows = []
    if not args.skip_external:
        mendeley_rows = read_subject_feature_csv(args.mendeley_features, dataset_name="mendeley")
        ds007216_rows = read_subject_feature_csv(args.ds007216_features, dataset_name="ds007216")

    for seed in args.seeds:
        train_ds, val_ds, split_info = build_joint_datasets(
            dataset_csvs=dataset_csvs,
            seed=int(seed),
            val_fraction=0.30,
            gray_z=args.gray_z,
            gray_zone_weight=args.gray_zone_weight,
            threshold_mode=args.threshold_mode,
            feature_preset=args.feature_preset,
            include_exploratory_features=False,
        )

        x_train = np.stack([sample.x for sample in train_ds.samples], axis=0)
        y_train = np.asarray([sample.label for sample in train_ds.samples], dtype=np.int64)
        x_val = np.stack([sample.x for sample in val_ds.samples], axis=0)
        y_val = np.asarray([sample.label for sample in val_ds.samples], dtype=np.int64)
        val_gray = np.asarray([sample.in_gray_zone for sample in val_ds.samples], dtype=bool)

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=int(seed),
        )
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        val_metrics = binary_metrics(y_val, val_prob, in_gray_zone=val_gray)
        val_metrics["joint_score"] = selection_score(val_metrics)
        internal_rows.append(
            {
                "seed": int(seed),
                "model_name": "shared_subspace_logreg",
                **val_metrics,
            }
        )
        for sample, prob in zip(val_ds.samples, val_prob):
            internal_prediction_rows.append(
                {
                    "seed": int(seed),
                    "dataset": sample.dataset,
                    "subject": sample.subject,
                    "anxiety": float(sample.anxiety),
                    "label": int(sample.label),
                    "in_gray_zone": bool(sample.in_gray_zone),
                    "prob_high": float(prob),
                }
            )

        if not args.skip_external:
            feature_names = list(split_info["input_features"])
            means, stds = pooled_normalizer(split_info)

            x_mendeley = build_external_matrix(mendeley_rows, feature_names, means, stds)
            y_mendeley = np.asarray([int(round(float(row.anxiety))) for row in mendeley_rows], dtype=np.int64)
            prob_mendeley = model.predict_proba(x_mendeley)[:, 1]
            mendeley_metrics = binary_metrics(y_mendeley, prob_mendeley)
            mendeley_summary_rows.append(
                {
                    "seed": int(seed),
                    "model_name": "shared_subspace_logreg",
                    **mendeley_metrics,
                }
            )
            for row, prob in zip(mendeley_rows, prob_mendeley):
                mendeley_prediction_rows.append(
                    {
                        "seed": int(seed),
                        "subject": row.subject,
                        "anxiety": float(row.anxiety),
                        "label": int(round(float(row.anxiety))),
                        "prob_high": float(prob),
                    }
                )

            thresholds = compute_external_thresholds(ds007216_rows, gray_z=args.gray_z)
            x_ds007216 = build_external_matrix(ds007216_rows, feature_names, means, stds)
            y_ds007216 = np.asarray([int(float(row.anxiety) >= thresholds["center"]) for row in ds007216_rows], dtype=np.int64)
            gray_ds007216 = np.asarray(
                [thresholds["low_threshold"] < float(row.anxiety) < thresholds["high_threshold"] for row in ds007216_rows],
                dtype=bool,
            )
            prob_ds007216 = model.predict_proba(x_ds007216)[:, 1]
            ds007216_metrics = binary_metrics(y_ds007216, prob_ds007216, in_gray_zone=gray_ds007216)
            ds007216_metrics["flip_roc_auc"] = safe_roc_auc(y_ds007216, 1.0 - prob_ds007216)
            ds007216_summary_rows.append(
                {
                    "seed": int(seed),
                    "model_name": "shared_subspace_logreg",
                    **ds007216_metrics,
                }
            )
            for row, label, gray, prob in zip(ds007216_rows, y_ds007216, gray_ds007216, prob_ds007216):
                ds007216_prediction_rows.append(
                    {
                        "seed": int(seed),
                        "subject": row.subject,
                        "anxiety": float(row.anxiety),
                        "label": int(label),
                        "in_gray_zone": bool(gray),
                        "prob_high": float(prob),
                    }
                )

    write_csv(output_root / "internal_seed_results.csv", internal_rows)
    write_csv(output_root / "internal_predictions.csv", internal_prediction_rows)
    write_csv(output_root / "mendeley_seed_results.csv", mendeley_summary_rows)
    write_csv(output_root / "mendeley_predictions.csv", mendeley_prediction_rows)
    write_csv(output_root / "ds007216_seed_results.csv", ds007216_summary_rows)
    write_csv(output_root / "ds007216_predictions.csv", ds007216_prediction_rows)

    metric_names = ["joint_score", "accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc", "extreme_balanced_accuracy", "extreme_roc_auc"]
    write_csv(output_root / "internal_aggregate.csv", aggregate_mean_std(internal_rows, metric_names))
    write_csv(output_root / "mendeley_aggregate.csv", aggregate_mean_std(mendeley_summary_rows, ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]))
    write_csv(output_root / "ds007216_aggregate.csv", aggregate_mean_std(ds007216_summary_rows, ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc", "flip_roc_auc"]))

    write_json(
        output_root / "run_config.json",
        {
            "seeds": [int(seed) for seed in args.seeds],
            "feature_preset": args.feature_preset,
            "threshold_mode": args.threshold_mode,
            "gray_z": float(args.gray_z),
            "gray_zone_weight": float(args.gray_zone_weight),
            "features_root": str(Path(args.features_root)),
            "skip_external": bool(args.skip_external),
            "feature_names": list(
                build_joint_datasets(
                    dataset_csvs=dataset_csvs,
                    feature_preset=args.feature_preset,
                )[2]["input_features"]
            ),
        },
    )
    print(f"Saved shared-subspace logistic outputs to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
