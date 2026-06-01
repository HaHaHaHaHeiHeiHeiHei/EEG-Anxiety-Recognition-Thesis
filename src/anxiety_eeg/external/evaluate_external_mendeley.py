"""中文说明

用途：
    加载主模型 checkpoint，在 Mendeley anxiety/control 数据上做外部兼容性验证。
输入：
    `--train-output-root` 为主模型训练输出，`--scoring-dir` 含 Mendeley
    `subject_features.csv`，或提供 `--workbook` 自动提取。
输出：
    外部预测、seed-level summary、aggregate JSON/CSV 和 Markdown 报告。
快速运行：
    `python -m anxiety_eeg.external.evaluate_external_mendeley --train-output-root outputs/joint_constraint --skip-scoring`
论文对应：
    第 5 章 Mendeley partial-overlap compatibility validation。
注意事项：
    缺失训练特征会按训练集均值中性填补，因此结果只能解释为兼容性验证。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


from anxiety_eeg.config import apply_json_config
from anxiety_eeg.models.joint_constraint import JointConstraintNet  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_OUTPUT_ROOT = REPO_ROOT / "outputs" / "joint_constraint"
DEFAULT_SCORING_DIR = REPO_ROOT / "features" / "external" / "mendeley"
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "external_mendeley"
DEFAULT_WORKBOOK = REPO_ROOT / "data" / "mendeley_anxiety_control" / "EEG_data.xlsx"
DEFAULT_EXTRACT_SCRIPT = REPO_ROOT / "src" / "anxiety_eeg" / "external" / "extract_mendeley_subject_features.py"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load trained joint-constraint checkpoints and evaluate external compatibility generalization on the Mendeley anxiety/control workbook."
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config file; keys match CLI argument names.")
    parser.add_argument("--train-output-root", type=Path, default=DEFAULT_TRAIN_OUTPUT_ROOT)
    parser.add_argument("--scoring-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--extract-script", type=Path, default=DEFAULT_EXTRACT_SCRIPT)
    parser.add_argument(
        "--condition",
        choices=["observ", "imagin", "execut", "together", "mean_tasks"],
        default="together",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["shared_only", "mean_adapter", "proxy_dataset"],
        default=["shared_only", "mean_adapter"],
    )
    parser.add_argument(
        "--proxy-dataset",
        choices=["original_local", "ds003478", "ds007609"],
        default="ds007609",
    )
    parser.add_argument("--gray-z", type=float, default=0.35)
    parser.add_argument("--rerun-scoring", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return apply_json_config(parser.parse_args())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_feature_workbook(subject_features_path: Path, requested_workbook: Path) -> str:
    dataset_info_path = subject_features_path.parent / "dataset_info.json"
    if dataset_info_path.exists():
        try:
            dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
            extracted_workbook = dataset_info.get("config", {}).get("workbook")
            if extracted_workbook:
                return str(Path(extracted_workbook).resolve())
        except (OSError, TypeError, json.JSONDecodeError):
            pass
    return str(requested_workbook.resolve())


def finite_float(value: object) -> float:
    out = float(str(value).strip())
    if not math.isfinite(out):
        raise ValueError(f"Non-finite float value: {value!r}")
    return out


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


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


def binary_metrics(labels: np.ndarray, probs: np.ndarray, in_gray_zone: np.ndarray) -> dict[str, float | list[list[int]]]:
    pred = (probs >= 0.5).astype(np.int64)
    metrics: dict[str, float | list[list[int]]] = {
        "n_subjects": int(len(labels)),
        "n_low": int(np.sum(labels == 0)),
        "n_high": int(np.sum(labels == 1)),
        "accuracy": float(accuracy_score(labels, pred)) if len(np.unique(labels)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)) if len(np.unique(labels)) > 1 else float("nan"),
        "f1": float(f1_score(labels, pred, zero_division=0)) if len(np.unique(labels)) > 1 else float("nan"),
        "roc_auc": safe_roc_auc(labels, probs),
        "pr_auc": safe_pr_auc(labels, probs),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
        "gray_subjects": int(np.sum(in_gray_zone)),
    }
    extreme_mask = ~in_gray_zone
    if np.sum(extreme_mask) >= 3 and len(np.unique(labels[extreme_mask])) > 1:
        metrics["extreme_accuracy"] = float(accuracy_score(labels[extreme_mask], pred[extreme_mask]))
        metrics["extreme_balanced_accuracy"] = float(balanced_accuracy_score(labels[extreme_mask], pred[extreme_mask]))
        metrics["extreme_f1"] = float(f1_score(labels[extreme_mask], pred[extreme_mask], zero_division=0))
        metrics["extreme_roc_auc"] = safe_roc_auc(labels[extreme_mask], probs[extreme_mask])
        metrics["extreme_pr_auc"] = safe_pr_auc(labels[extreme_mask], probs[extreme_mask])
    else:
        metrics["extreme_accuracy"] = float("nan")
        metrics["extreme_balanced_accuracy"] = float("nan")
        metrics["extreme_f1"] = float("nan")
        metrics["extreme_roc_auc"] = float("nan")
        metrics["extreme_pr_auc"] = float("nan")
    return metrics


def is_numeric_scalar(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def aggregate_numeric_rows(rows: list[dict], group_key: str | None, exclude_metrics: set[str] | None = None) -> list[dict]:
    exclude_metrics = set(exclude_metrics or set())
    grouped: dict[object, list[dict]] = defaultdict(list)
    if group_key is None:
        grouped["overall"] = list(rows)
    else:
        for row in rows:
            grouped[row[group_key]].append(row)
    out_rows = []
    for group_value, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
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
            arr = np.array(values, dtype=np.float64)
            out_rows.append(
                {
                    "scope": "overall" if group_key is None else str(group_key),
                    "group": "all" if group_key is None else str(group_value),
                    "metric": metric_name,
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "median": float(np.median(arr)),
                    "max": float(np.max(arr)),
                }
            )
    return out_rows


def aggregate_payload(rows: list[dict], strategies: list[str]) -> tuple[list[dict], dict]:
    aggregate_rows = aggregate_numeric_rows(rows, group_key="strategy", exclude_metrics={"seed"})
    payload = {"strategies": list(strategies), "by_strategy": {}}
    for row in aggregate_rows:
        strategy = row["group"]
        payload["by_strategy"].setdefault(strategy, {})
        payload["by_strategy"][strategy][row["metric"]] = {
            "n": row["n"],
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "median": row["median"],
            "max": row["max"],
        }
    return aggregate_rows, payload


def run_logged_subprocess(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"\n[exit_code] {process.returncode}\n")
        handle.write(f"[elapsed_sec] {time.time() - started:.1f}\n")
    if process.returncode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {process.returncode}. See log: {log_path}")


def ensure_mendeley_scoring(args: argparse.Namespace) -> Path:
    subject_features_path = args.scoring_dir / "subject_features.csv"
    if args.skip_scoring and not subject_features_path.exists():
        raise FileNotFoundError(f"--skip-scoring was set but {subject_features_path} does not exist")
    if subject_features_path.exists() and not args.rerun_scoring:
        return subject_features_path

    args.scoring_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(args.extract_script),
        "--workbook",
        str(args.workbook),
        "--output-dir",
        str(args.scoring_dir),
        "--condition",
        str(args.condition),
    ]
    run_logged_subprocess(
        command=command,
        cwd=REPO_ROOT,
        log_path=args.scoring_dir / "extract_mendeley.log",
    )
    if not subject_features_path.exists():
        raise FileNotFoundError(f"Expected subject_features.csv after Mendeley extraction: {subject_features_path}")
    return subject_features_path


def read_external_subject_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No rows found in external subject feature csv: {path}")
    return rows


def compute_external_thresholds(rows: list[dict], gray_z: float) -> dict:
    scores = np.array([finite_float(row["anxiety"]) for row in rows], dtype=np.float32)
    unique = sorted(set(float(value) for value in scores.tolist()))
    if unique == [0.0, 1.0]:
        return {
            "mode": "binary_fixed",
            "center": 0.5,
            "std": 0.5,
            "gray_z": 0.0,
            "low_threshold": 0.49,
            "high_threshold": 0.51,
        }
    center = float(np.median(scores))
    std = float(np.std(scores))
    if not math.isfinite(std) or std < EPS:
        std = 1.0
    return {
        "mode": "continuous_median_std",
        "center": center,
        "std": std,
        "gray_z": float(gray_z),
        "low_threshold": float(center - float(gray_z) * std),
        "high_threshold": float(center + float(gray_z) * std),
    }


def pooled_normalizer(config: dict) -> tuple[dict[str, float], dict[str, float]]:
    feature_names = list(config["input_features"])
    per_dataset = dict(config["per_dataset"])
    counts = {name: int(info["train_count"]) for name, info in per_dataset.items()}
    total = float(sum(counts.values()))
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature_name in feature_names:
        pooled_mean = sum(
            counts[name] * float(per_dataset[name]["normalizer"]["feature_means"][feature_name])
            for name in per_dataset
        ) / max(total, 1.0)
        pooled_var = sum(
            counts[name]
            * (
                float(per_dataset[name]["normalizer"]["feature_stds"][feature_name]) ** 2
                + (
                    float(per_dataset[name]["normalizer"]["feature_means"][feature_name]) - pooled_mean
                )
                ** 2
            )
            for name in per_dataset
        ) / max(total, 1.0)
        means[feature_name] = float(pooled_mean)
        stds[feature_name] = float(max(math.sqrt(max(pooled_var, EPS)), 1.0e-6))
    return means, stds


def build_external_matrix_with_imputation(
    rows: list[dict],
    feature_names: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> tuple[np.ndarray, dict[str, int]]:
    missing_counts = {feature_name: 0 for feature_name in feature_names}
    out = []
    for row in rows:
        values = []
        for feature_name in feature_names:
            if feature_name in row:
                try:
                    raw = finite_float(row[feature_name])
                except Exception:
                    raw = means[feature_name]
                    missing_counts[feature_name] += 1
            else:
                raw = means[feature_name]
                missing_counts[feature_name] += 1
            values.append(float((raw - means[feature_name]) / (stds[feature_name] + EPS)))
        out.append(values)
    return np.asarray(out, dtype=np.float32), missing_counts


def build_model_from_config(config: dict, device: torch.device) -> JointConstraintNet:
    model = JointConstraintNet(
        input_dim=len(config["input_features"]),
        dataset_count=len(config["dataset_order"]),
        global_dim=len(config["global_constraint_features"]),
        region_dim=len(config["region_constraint_features"]),
        hidden_dim=int(config["hidden_dim"]),
        adapter_dim=int(config["adapter_dim"]),
        dropout=float(config["dropout"]),
    )
    return model.to(device)


def load_checkpoint(model: JointConstraintNet, checkpoint_path: Path, device: torch.device) -> None:
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


def resolve_checkpoint_path(seed_dir: Path, config: dict) -> Path:
    raw = str(config.get("final_summary", {}).get("checkpoint", "")).strip()
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        candidate = REPO_ROOT / candidate
        if candidate.exists():
            return candidate
    fallback = seed_dir / "best_joint_constraint.pt"
    if not fallback.exists():
        raise FileNotFoundError(f"Checkpoint not found for seed_dir={seed_dir}")
    return fallback


@torch.no_grad()
def predict_external(
    model: JointConstraintNet,
    x: np.ndarray,
    strategy: str,
    dataset_order: list[str],
    proxy_dataset: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    shared = model.shared_trunk(model.stem(model.input_norm(x_tensor)))

    hidden_dim = int(model.classifier.net[0].in_features)
    if strategy == "shared_only":
        adapter_hidden = torch.zeros(hidden_dim, device=device)
        bias = torch.tensor(0.0, device=device)
    elif strategy == "mean_adapter":
        embedding = model.dataset_embedding.weight.detach().mean(dim=0, keepdim=True)
        adapter_hidden = model.adapter(embedding).squeeze(0).to(device)
        bias = model.dataset_bias.weight.detach().mean().to(device)
    elif strategy == "proxy_dataset":
        if proxy_dataset not in dataset_order:
            raise KeyError(f"proxy_dataset={proxy_dataset!r} not in dataset_order={dataset_order}")
        proxy_index = int(dataset_order.index(proxy_dataset))
        embedding = model.dataset_embedding.weight.detach()[proxy_index : proxy_index + 1]
        adapter_hidden = model.adapter(embedding).squeeze(0).to(device)
        bias = model.dataset_bias.weight.detach()[proxy_index, 0].to(device)
    else:
        raise ValueError(f"Unsupported strategy={strategy!r}")

    adapted = shared + adapter_hidden.unsqueeze(0)
    logits = model.classifier(adapted).squeeze(-1) + bias
    probs = torch.sigmoid(logits)
    return logits.detach().cpu().numpy(), probs.detach().cpu().numpy()


def evaluate_one_seed(
    seed_dir: Path,
    external_rows: list[dict],
    thresholds: dict,
    strategies: list[str],
    proxy_dataset: str,
    device: torch.device,
) -> tuple[list[dict], dict[str, list[dict]], dict]:
    config = json.loads((seed_dir / "config_joint_constraint.json").read_text(encoding="utf-8"))
    checkpoint_path = resolve_checkpoint_path(seed_dir, config)
    model = build_model_from_config(config, device)
    load_checkpoint(model, checkpoint_path, device)

    means, stds = pooled_normalizer(config)
    x, imputation_counts = build_external_matrix_with_imputation(
        external_rows,
        list(config["input_features"]),
        means,
        stds,
    )

    anxiety = np.asarray([finite_float(row["anxiety"]) for row in external_rows], dtype=np.float32)
    labels = (anxiety >= float(thresholds["center"])).astype(np.int64)
    if thresholds.get("mode") == "binary_fixed":
        in_gray_zone = np.zeros_like(labels, dtype=bool)
    else:
        in_gray_zone = np.logical_and(
            anxiety > float(thresholds["low_threshold"]),
            anxiety < float(thresholds["high_threshold"]),
        )

    summary_rows = []
    prediction_rows_by_strategy: dict[str, list[dict]] = {}

    for strategy in strategies:
        logits, probs = predict_external(
            model=model,
            x=x,
            strategy=strategy,
            dataset_order=list(config["dataset_order"]),
            proxy_dataset=proxy_dataset,
            device=device,
        )
        metrics = binary_metrics(labels=labels, probs=probs, in_gray_zone=in_gray_zone)
        summary_rows.append(
            {
                "seed": int(config["seed"]),
                "strategy": strategy,
                "checkpoint": str(checkpoint_path),
                "score_name": str(external_rows[0].get("score_name", "")),
                "n_subjects": int(metrics["n_subjects"]),
                "gray_subjects": int(metrics["gray_subjects"]),
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
                "imputed_feature_count": int(sum(imputation_counts.values())),
                "imputed_feature_names": ";".join(sorted(name for name, count in imputation_counts.items() if count > 0)),
            }
        )
        prediction_rows_by_strategy[strategy] = [
            {
                "seed": int(config["seed"]),
                "strategy": strategy,
                "subject": row.get("subject", ""),
                "context": row.get("context", ""),
                "anxiety": float(anxiety[idx]),
                "label": int(labels[idx]),
                "in_gray_zone": bool(in_gray_zone[idx]),
                "logit": float(logits[idx]),
                "prob_high": float(probs[idx]),
                "pred_label": int(float(probs[idx]) >= 0.5),
            }
            for idx, row in enumerate(external_rows)
        ]

    run_meta = {
        "seed": int(config["seed"]),
        "checkpoint": str(checkpoint_path),
        "thresholds": thresholds,
        "imputation_counts": {key: int(value) for key, value in sorted(imputation_counts.items()) if value > 0},
        "input_features": list(config["input_features"]),
    }
    return summary_rows, prediction_rows_by_strategy, run_meta


def build_markdown_summary(payload: dict, rows: list[dict], config: dict) -> str:
    lines = [
        "# Mendeley External Compatibility Summary",
        "",
        f"- Train root: `{config['train_output_root']}`",
        f"- Workbook: `{config['workbook']}`",
        f"- External scoring dir: `{config['scoring_dir']}`",
        f"- Condition: `{config['condition']}`",
        f"- Threshold mode: `{payload['external_thresholds'].get('mode', 'unknown')}`",
        f"- Proxy dataset: `{config['proxy_dataset']}`",
        "",
        "## Compatibility Notes",
        "",
        "- This dataset is evaluated as binary `patients/control`, not continuous anxiety regression-derived splitting.",
        "- Missing checkpoint-required EEG features are imputed to the pooled training mean, so their normalized value is 0.",
        "- Interpret this as a compatibility external test, not a strict full-feature-space match.",
        "",
        "## Strategy Aggregate",
        "",
        "| Strategy | Balanced Accuracy | ROC-AUC | Extreme Balanced Accuracy | Extreme ROC-AUC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for strategy, metrics in payload["by_strategy"].items():
        lines.append(
            "| "
            + strategy
            + " | "
            + f"{metrics['balanced_accuracy']['mean']:.3f} +/- {metrics['balanced_accuracy']['std']:.3f}"
            + " | "
            + f"{metrics['roc_auc']['mean']:.3f} +/- {metrics['roc_auc']['std']:.3f}"
            + " | "
            + f"{metrics['extreme_balanced_accuracy']['mean']:.3f} +/- {metrics['extreme_balanced_accuracy']['std']:.3f}"
            + " | "
            + f"{metrics['extreme_roc_auc']['mean']:.3f} +/- {metrics['extreme_roc_auc']['std']:.3f}"
            + " |"
        )

    lines.extend(
        [
            "",
            "## Seed Table",
            "",
            "| Seed | Strategy | BalAcc | AUC | Imputed Features |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["strategy"]), int(item["seed"]))):
        lines.append(
            f"| {row['seed']} | {row['strategy']} | "
            f"{float(row['balanced_accuracy']):.3f} | {float(row['roc_auc']):.3f} | "
            f"{row['imputed_feature_names'] or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    subject_features_path = ensure_mendeley_scoring(args)
    external_rows = read_external_subject_rows(subject_features_path)
    thresholds = compute_external_thresholds(external_rows, gray_z=args.gray_z)

    summary_rows_all: list[dict] = []
    run_meta_rows: list[dict] = []
    prediction_root = args.results_dir / "predictions"

    seed_dirs = sorted(
        [path for path in args.train_output_root.iterdir() if path.is_dir() and path.name.startswith("seed_")],
        key=lambda path: int(path.name.split("_", 1)[1]),
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* folders found under train_output_root={args.train_output_root}")

    started = time.time()
    for seed_dir in seed_dirs:
        summary_rows, prediction_rows_by_strategy, run_meta = evaluate_one_seed(
            seed_dir=seed_dir,
            external_rows=external_rows,
            thresholds=thresholds,
            strategies=list(args.strategies),
            proxy_dataset=args.proxy_dataset,
            device=device,
        )
        summary_rows_all.extend(summary_rows)
        run_meta_rows.append(run_meta)
        for strategy, prediction_rows in prediction_rows_by_strategy.items():
            write_csv(
                prediction_root / f"{seed_dir.name}_{strategy}_predictions_mendeley.csv",
                prediction_rows,
            )

    aggregate_rows, payload = aggregate_payload(summary_rows_all, list(args.strategies))
    runtime_sec = float(time.time() - started)
    feature_workbook = resolve_feature_workbook(subject_features_path, args.workbook)
    requested_workbook = str(args.workbook.resolve())
    payload["runtime_sec"] = runtime_sec
    payload["n_seeds"] = int(len(seed_dirs))
    payload["seed_values"] = [int(path.name.split("_", 1)[1]) for path in seed_dirs]
    payload["external_thresholds"] = thresholds
    payload["train_output_root"] = str(args.train_output_root.resolve())
    payload["scoring_dir"] = str(args.scoring_dir.resolve())
    payload["workbook"] = feature_workbook
    if feature_workbook != requested_workbook:
        payload["workbook_requested"] = requested_workbook
    payload["condition"] = str(args.condition)
    payload["proxy_dataset"] = str(args.proxy_dataset)
    payload["overall_imputed_features"] = sorted(
        {
            name
            for row in summary_rows_all
            for name in str(row.get("imputed_feature_names", "")).split(";")
            if name
        }
    )

    write_csv(args.results_dir / "summary_all_seeds_mendeley.csv", summary_rows_all)
    write_csv(args.results_dir / "summary_aggregate_mendeley.csv", aggregate_rows)
    write_json(args.results_dir / "summary_aggregate_mendeley.json", payload)
    write_json(args.results_dir / "run_meta_mendeley.json", {"rows": run_meta_rows})
    write_text(
        args.results_dir / "summary_mendeley.md",
        build_markdown_summary(
            payload=payload,
            rows=summary_rows_all,
            config={
                "train_output_root": str(args.train_output_root),
                "scoring_dir": str(args.scoring_dir),
                "workbook": feature_workbook,
                "condition": str(args.condition),
                "proxy_dataset": str(args.proxy_dataset),
            },
        ),
    )

    print(f"[Device] {device}")
    print(f"[Mendeley] subject_features={subject_features_path}")
    print(f"[Mendeley] feature_workbook={feature_workbook}")
    print(f"[Mendeley] results_dir={args.results_dir.resolve()}")
    for strategy in args.strategies:
        metrics = payload["by_strategy"].get(strategy, {})
        bal = metrics.get("balanced_accuracy")
        auc = metrics.get("roc_auc")
        if not bal or not auc:
            continue
        print(
            f"[Strategy:{strategy}] "
            f"bal={bal['mean']:.4f}+/-{bal['std']:.4f} "
            f"auc={auc['mean']:.4f}+/-{auc['std']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
