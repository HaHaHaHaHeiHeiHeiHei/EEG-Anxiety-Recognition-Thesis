"""
中文说明
用途：训练论文对照实验中的传统机器学习基线，包括 LogReg-L2、SVM、随机森林、
Extra Trees、Gradient Boosting，以及显式指定时才运行的可选 XGBoost。

输入：`--features-root` 指向受试者级 EEG 特征根目录，结构与主模型相同。
输出：每个 seed/模型的验证指标、预测文件和聚合 summary，默认写入
`outputs/traditional_baselines`。

快速运行：
`python scripts/train_baselines.py --features-root tests/fixtures/subject_features --seeds 42 --models logreg_l2 --n-jobs 1`

论文对应：第 5 章“主模型与传统基线比较”。
注意事项：
- 基线复用与主模型一致的 subject-level split、训练集阈值和 gray-zone 权重。
- 默认基线使用固定超参数；不应在论文中写成 train-only CV，除非另行实现内层交叉验证。
- `--models all` 只运行默认基线，不再默认触发 xgboost 依赖。
- 如需运行 XGBoost，请先安装 xgboost，并显式传入 `--models xgboost`。
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from anxiety_eeg.config import apply_json_config
from anxiety_eeg.data.joint_dataset import (
    DEFAULT_FEATURES_ROOT,
    build_joint_datasets,
    dataset_csvs_from_root,
    weighted_binary_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "traditional_baselines"

DEFAULT_SEEDS = [
    42,
    123,
    224,
    3407,
    65422,
    7,
    21,
    2024,
    31415,
    27182,
    97,
    512,
    777,
    1024,
    2048,
    4096,
    8192,
    11111,
    12345,
    13579,
    24680,
    50001,
    60013,
    70001,
    88888,
]

# 论文表 5.1 的默认传统基线。保持无额外依赖，便于本科论文复现。
DEFAULT_MODELS = [
    "logreg_l2",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
]

# XGBoost 作为可选扩展模型：显式指定 --models xgboost 时才运行。
OPTIONAL_MODELS = ["xgboost"]
AVAILABLE_MODELS = DEFAULT_MODELS + OPTIONAL_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traditional fixed-parameter baselines aligned to the joint constraint model: "
            "same subject split, gray-zone labels/weights, 25 seeds, and input features. "
            "Use --models all for default dependency-light baselines; use --models xgboost "
            "only after installing xgboost."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config file; keys match CLI argument names.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--gray-z", type=float, default=0.35)
    parser.add_argument("--gray-zone-weight", type=float, default=0.35)
    parser.add_argument("--include-exploratory-features", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return apply_json_config(parser.parse_args())


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def finite(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def is_numeric_scalar(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    in_gray_zone: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    pred = (scores >= float(threshold)).astype(np.int64)
    metrics: dict[str, float | list[list[int]]] = {
        "n_subjects": int(len(labels)),
        "n_low": int(np.sum(labels == 0)),
        "n_high": int(np.sum(labels == 1)),
        "accuracy": float(accuracy_score(labels, pred)) if len(np.unique(labels)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)) if len(np.unique(labels)) > 1 else float("nan"),
        "f1": float(f1_score(labels, pred, zero_division=0)) if len(np.unique(labels)) > 1 else float("nan"),
        "roc_auc": safe_roc_auc(labels, scores),
        "pr_auc": safe_pr_auc(labels, scores),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
        "gray_subjects": int(np.sum(in_gray_zone)),
    }

    extreme_mask = ~in_gray_zone
    if np.sum(extreme_mask) >= 3 and len(np.unique(labels[extreme_mask])) > 1:
        metrics["extreme_accuracy"] = float(accuracy_score(labels[extreme_mask], pred[extreme_mask]))
        metrics["extreme_balanced_accuracy"] = float(balanced_accuracy_score(labels[extreme_mask], pred[extreme_mask]))
        metrics["extreme_f1"] = float(f1_score(labels[extreme_mask], pred[extreme_mask], zero_division=0))
        metrics["extreme_roc_auc"] = safe_roc_auc(labels[extreme_mask], scores[extreme_mask])
        metrics["extreme_pr_auc"] = safe_pr_auc(labels[extreme_mask], scores[extreme_mask])
    else:
        metrics["extreme_accuracy"] = float("nan")
        metrics["extreme_balanced_accuracy"] = float("nan")
        metrics["extreme_f1"] = float("nan")
        metrics["extreme_roc_auc"] = float("nan")
        metrics["extreme_pr_auc"] = float("nan")
    return metrics


def selection_score(metrics: dict[str, float]) -> float:
    return (
        0.35 * max(0.0, finite(metrics.get("extreme_roc_auc")))
        + 0.25 * max(0.0, finite(metrics.get("extreme_balanced_accuracy")))
        + 0.25 * max(0.0, finite(metrics.get("roc_auc")))
        + 0.15 * max(0.0, finite(metrics.get("balanced_accuracy")))
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, ensure_ascii=False, indent=2)


def dataset_to_arrays(dataset) -> dict[str, np.ndarray]:
    return {
        "x": np.stack([sample.x for sample in dataset.samples], axis=0).astype(np.float32),
        "label": np.asarray([sample.label for sample in dataset.samples], dtype=np.int64),
        "sample_weight": np.asarray([sample.sample_weight for sample in dataset.samples], dtype=np.float64),
        "in_gray_zone": np.asarray([sample.in_gray_zone for sample in dataset.samples], dtype=bool),
        "dataset": np.asarray([sample.dataset for sample in dataset.samples], dtype=object),
        "subject": np.asarray([sample.subject for sample in dataset.samples], dtype=object),
        "subject_uid": np.asarray([sample.subject_uid for sample in dataset.samples], dtype=object),
        "source_split": np.asarray([sample.source_split for sample in dataset.samples], dtype=object),
        "score_name": np.asarray([sample.score_name for sample in dataset.samples], dtype=object),
        "context": np.asarray([sample.context for sample in dataset.samples], dtype=object),
        "anxiety": np.asarray([sample.anxiety for sample in dataset.samples], dtype=np.float64),
    }


def effective_fit_weights(labels: np.ndarray, sample_weight: np.ndarray, pos_weight_scalar: float) -> np.ndarray:
    class_factor = np.where(labels.astype(np.int64) == 1, float(pos_weight_scalar), 1.0)
    return sample_weight.astype(np.float64) * class_factor


def build_estimator(model_name: str, seed: int, n_jobs: int):
    if model_name == "logreg_l2":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=int(seed),
        )
    if model_name == "linear_svm":
        return SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=int(seed),
            cache_size=500,
        )
    if model_name == "rbf_svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=int(seed),
            cache_size=500,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            random_state=int(seed),
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "Model 'xgboost' was requested explicitly, but xgboost is not installed. "
                "Install it first, or use --models all for the default dependency-light baselines."
            ) from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def predict_scores(estimator, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=np.float64)
        return np.asarray(proba, dtype=np.float64).reshape(-1)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(x), dtype=np.float64).reshape(-1)
        return 1.0 / (1.0 + np.exp(-raw))
    pred = np.asarray(estimator.predict(x), dtype=np.float64).reshape(-1)
    return pred


def metrics_by_dataset(arrays: dict[str, np.ndarray], scores: np.ndarray, model_name: str, seed: int) -> list[dict]:
    rows = []
    labels = arrays["label"]
    pred = (scores >= 0.5).astype(np.int64)
    for dataset_name in sorted(set(arrays["dataset"].tolist())):
        mask = arrays["dataset"] == dataset_name
        metrics = binary_metrics(labels[mask], scores[mask], arrays["in_gray_zone"][mask])
        rows.append(
            {
                "seed": int(seed),
                "model_name": model_name,
                "dataset": dataset_name,
                "n_subjects": metrics["n_subjects"],
                "n_low": metrics["n_low"],
                "n_high": metrics["n_high"],
                "gray_subjects": metrics["gray_subjects"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "extreme_accuracy": metrics["extreme_accuracy"],
                "extreme_balanced_accuracy": metrics["extreme_balanced_accuracy"],
                "extreme_f1": metrics["extreme_f1"],
                "extreme_roc_auc": metrics["extreme_roc_auc"],
                "extreme_pr_auc": metrics["extreme_pr_auc"],
                "pred_low": int(np.sum(pred[mask] == 0)),
                "pred_high": int(np.sum(pred[mask] == 1)),
            }
        )
    return rows


def prediction_rows(arrays: dict[str, np.ndarray], scores: np.ndarray, model_name: str, seed: int) -> list[dict]:
    pred = (scores >= 0.5).astype(np.int64)
    rows = []
    for index in range(len(scores)):
        rows.append(
            {
                "seed": int(seed),
                "model_name": model_name,
                "dataset": str(arrays["dataset"][index]),
                "subject": str(arrays["subject"][index]),
                "subject_uid": str(arrays["subject_uid"][index]),
                "label": int(arrays["label"][index]),
                "pred": int(pred[index]),
                "score_high": float(scores[index]),
                "anxiety": float(arrays["anxiety"][index]),
                "sample_weight": float(arrays["sample_weight"][index]),
                "in_gray_zone": bool(arrays["in_gray_zone"][index]),
                "source_split": str(arrays["source_split"][index]),
            }
        )
    return rows


def train_eval_model(
    model_name: str,
    seed: int,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    pos_weight_scalar: float,
    n_jobs: int,
) -> tuple[dict, list[dict], list[dict]]:
    estimator = build_estimator(model_name=model_name, seed=seed, n_jobs=n_jobs)
    fit_weight = effective_fit_weights(
        labels=train_arrays["label"],
        sample_weight=train_arrays["sample_weight"],
        pos_weight_scalar=pos_weight_scalar,
    )

    started = time.time()
    estimator.fit(train_arrays["x"], train_arrays["label"], sample_weight=fit_weight)
    fit_seconds = time.time() - started

    scores = predict_scores(estimator, val_arrays["x"])
    metrics = binary_metrics(val_arrays["label"], scores, val_arrays["in_gray_zone"])
    joint_score = selection_score(metrics)
    dataset_rows = metrics_by_dataset(val_arrays, scores, model_name=model_name, seed=seed)
    pred_rows = prediction_rows(val_arrays, scores, model_name=model_name, seed=seed)
    summary = {
        "seed": int(seed),
        "model_name": model_name,
        "fit_seconds": float(fit_seconds),
        "train_subjects": int(len(train_arrays["label"])),
        "val_subjects": int(len(val_arrays["label"])),
        "pos_weight": float(pos_weight_scalar),
        "selection_score": float(joint_score),
        "best_selection_score": float(joint_score),
        "best_val_balanced_accuracy": metrics["balanced_accuracy"],
        "best_val_roc_auc": metrics["roc_auc"],
        "best_val_extreme_balanced_accuracy": metrics["extreme_balanced_accuracy"],
        "best_val_extreme_roc_auc": metrics["extreme_roc_auc"],
        **metrics,
    }
    return summary, dataset_rows, pred_rows


def aggregate_numeric_rows(
    rows: list[dict],
    group_keys: list[str],
    exclude_metrics: set[str] | None = None,
) -> list[dict]:
    exclude_metrics = set(exclude_metrics or set())
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out_rows = []
    for group_value, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        metric_names = sorted(
            {
                key
                for row in group_rows
                for key, value in row.items()
                if key not in exclude_metrics and is_numeric_scalar(value)
            }
        )
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in group_rows
                if metric_name in row and is_numeric_scalar(row[metric_name]) and math.isfinite(float(row[metric_name]))
            ]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            out_row = {
                "metric": metric_name,
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "median": float(np.median(arr)),
                "max": float(np.max(arr)),
            }
            for key, value in zip(group_keys, group_value):
                out_row[key] = value
            out_rows.append(out_row)
    return out_rows


def build_aggregate_payload(
    seeds: list[int],
    summary_rows: list[dict],
    dataset_rows: list[dict],
) -> tuple[list[dict], dict]:
    summary_agg = aggregate_numeric_rows(
        summary_rows,
        group_keys=["model_name"],
        exclude_metrics={"seed", "confusion_matrix"},
    )
    dataset_agg = aggregate_numeric_rows(
        dataset_rows,
        group_keys=["model_name", "dataset"],
        exclude_metrics={"seed"},
    )
    payload: dict[str, Any] = {
        "n_seeds": int(len(seeds)),
        "seeds": [int(seed) for seed in seeds],
        "overall": {},
        "by_dataset": {},
    }
    for row in summary_agg:
        model_name = row["model_name"]
        payload["overall"].setdefault(model_name, {})
        payload["overall"][model_name][row["metric"]] = {
            "n": row["n"],
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "median": row["median"],
            "max": row["max"],
        }
    for row in dataset_agg:
        model_name = row["model_name"]
        dataset_name = row["dataset"]
        payload["by_dataset"].setdefault(model_name, {})
        payload["by_dataset"][model_name].setdefault(dataset_name, {})
        payload["by_dataset"][model_name][dataset_name][row["metric"]] = {
            "n": row["n"],
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "median": row["median"],
            "max": row["max"],
        }
    return summary_agg + dataset_agg, payload


def print_aggregate_report(payload: dict) -> None:
    print("\n========== traditional baseline aggregate ==========")
    print(f"[Seeds] n={payload.get('n_seeds')} values={payload.get('seeds')}")
    for model_name, metrics in sorted(payload.get("overall", {}).items()):
        joint = metrics.get("selection_score") or metrics.get("best_selection_score")
        bal = metrics.get("balanced_accuracy")
        auc = metrics.get("roc_auc")
        f1 = metrics.get("f1")
        if not (joint and bal and auc):
            continue
        print(
            f"[{model_name}] "
            f"joint={joint['mean']:.4f}+/-{joint['std']:.4f} "
            f"bal={bal['mean']:.4f}+/-{bal['std']:.4f} "
            f"auc={auc['mean']:.4f}+/-{auc['std']:.4f} "
            f"f1={f1['mean']:.4f}+/-{f1['std']:.4f}"
        )


def normalize_model_list(models: list[str]) -> list[str]:
    out: list[str] = []
    for name in models:
        if name == "all":
            for model_name in DEFAULT_MODELS:
                if model_name not in out:
                    out.append(model_name)
        else:
            if name not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model '{name}'. Available: {AVAILABLE_MODELS} or 'all'.")
            if name not in out:
                out.append(name)
    return out


def run_one_seed(args: argparse.Namespace, seed: int, model_names: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    set_seed(seed)
    dataset_csvs = dataset_csvs_from_root(args.features_root)
    train_ds, val_ds, split_info = build_joint_datasets(
        dataset_csvs=dataset_csvs,
        seed=int(seed),
        val_fraction=args.val_fraction,
        gray_z=args.gray_z,
        gray_zone_weight=args.gray_zone_weight,
        include_exploratory_features=args.include_exploratory_features,
    )
    train_arrays = dataset_to_arrays(train_ds)
    val_arrays = dataset_to_arrays(val_ds)
    neg_weight, pos_weight_value = weighted_binary_counts(train_ds)
    pos_weight_scalar = float(neg_weight / max(pos_weight_value, 1e-6))

    seed_dir = args.output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        seed_dir / "split_config_traditional_baselines.json",
        {
            "seed": int(seed),
            "features_root": str(Path(args.features_root)),
            "dataset_csvs": {name: str(path) for name, path in dataset_csvs.items()},
            "protocol": "matched_to_joint_constraint_model_baseline",
            "baseline_hyperparameter_rule": "fixed_hyperparameters_no_inner_cv",
            "split_rule": "per_dataset_subject_level_split",
            "threshold_rule": "train_set_median_per_dataset",
            "gray_zone_rule": "median +/- gray_z * train_std with down-weighted samples",
            "gray_z": float(args.gray_z),
            "gray_zone_weight": float(args.gray_zone_weight),
            "effective_fit_weight_rule": "sample_weight * pos_weight_for_positive_class",
            "pos_weight": pos_weight_scalar,
            "include_exploratory_features": bool(args.include_exploratory_features),
            **split_info,
        },
    )

    print(
        f"\n========== traditional baselines seed={seed} =========="
        f"\n[Data] train_subjects={len(train_ds)} val_subjects={len(val_ds)} datasets={split_info['dataset_order']}"
        f"\n[Weights] pos_weight={pos_weight_scalar:.4f} gray_weight={args.gray_zone_weight}"
    )

    summaries: list[dict] = []
    by_dataset: list[dict] = []
    predictions: list[dict] = []
    for model_name in model_names:
        print(f"[Model] seed={seed} model={model_name}")
        summary, dataset_rows, pred_rows = train_eval_model(
            model_name=model_name,
            seed=int(seed),
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            pos_weight_scalar=pos_weight_scalar,
            n_jobs=int(args.n_jobs),
        )
        summaries.append(summary)
        by_dataset.extend(dataset_rows)
        predictions.extend(pred_rows)
        model_dir = seed_dir / model_name
        write_csv(model_dir / f"summary_{model_name}.csv", [summary])
        write_csv(model_dir / f"val_metrics_by_dataset_{model_name}.csv", dataset_rows)
        write_csv(model_dir / f"val_subject_predictions_{model_name}.csv", pred_rows)
        print(
            f"[Result] {model_name} "
            f"joint={summary['selection_score']:.4f} "
            f"bal={finite(summary.get('balanced_accuracy')):.4f} "
            f"auc={finite(summary.get('roc_auc')):.4f} "
            f"f1={finite(summary.get('f1')):.4f}"
        )

    return summaries, by_dataset, predictions


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    model_names = normalize_model_list(args.models)
    print(f"[Output] {args.output_root.resolve()}")
    print(f"[Models] {model_names}")
    print(f"[Seeds] n={len(args.seeds)} values={[int(seed) for seed in args.seeds]}")

    all_summaries: list[dict] = []
    all_dataset_rows: list[dict] = []
    all_predictions: list[dict] = []
    for seed in args.seeds:
        summaries, dataset_rows, predictions = run_one_seed(args, int(seed), model_names)
        all_summaries.extend(summaries)
        all_dataset_rows.extend(dataset_rows)
        all_predictions.extend(predictions)

    write_csv(args.output_root / "summary_all_seeds_traditional_baselines.csv", all_summaries)
    write_csv(args.output_root / "summary_all_seeds_by_dataset_traditional_baselines.csv", all_dataset_rows)
    write_csv(args.output_root / "val_subject_predictions_all_seeds_traditional_baselines.csv", all_predictions)

    aggregate_rows, aggregate_payload = build_aggregate_payload(
        seeds=[int(seed) for seed in args.seeds],
        summary_rows=all_summaries,
        dataset_rows=all_dataset_rows,
    )
    write_csv(args.output_root / "summary_aggregate_traditional_baselines.csv", aggregate_rows)
    write_json(args.output_root / "summary_aggregate_traditional_baselines.json", aggregate_payload)
    print_aggregate_report(aggregate_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
