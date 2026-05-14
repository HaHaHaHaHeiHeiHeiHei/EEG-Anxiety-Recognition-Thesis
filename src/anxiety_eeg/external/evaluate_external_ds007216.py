"""中文说明

用途：
    加载主模型 checkpoint，在 ds007216 上进行严格外部 domain-shift 审计。
输入：
    `--train-output-root` 为主模型训练输出；`--scoring-dir` 为 ds007216
    `subject_features.csv` 所在目录；标签文件可由 `--label-file` 指定。
输出：
    外部预测、seed-level summary、aggregate JSON/CSV 和 Markdown 报告。
快速运行：
    `python -m anxiety_eeg.external.evaluate_external_ds007216 --train-output-root outputs/joint_constraint --skip-scoring`
论文对应：
    第 5 章 ds007216 外部敏感性和方向反转审计。
注意事项：
    ds007216 结果用于分析域偏移，不作为主模型性能优势声明。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
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
from anxiety_eeg.models.joint_constraint import JointConstraintNet


REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_OUTPUT_ROOT = REPO_ROOT / "outputs" / "joint_constraint"
DEFAULT_SCORING_DIR = REPO_ROOT / "features" / "external" / "ds007216"
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "external_ds007216"
DEFAULT_LABEL_FILE = REPO_ROOT / "features" / "external" / "ds007216_labels_stai_state_mean.tsv"
DEFAULT_DATA_CACHE_DIR = REPO_ROOT / "data" / "_cache" / "eegdash" / "ds007216"
DEFAULT_PARTICIPANTS_TSV = DEFAULT_DATA_CACHE_DIR / "participants.tsv"
DEFAULT_TEST_TOOL = REPO_ROOT / "src" / "anxiety_eeg" / "features" / "score_ds007216.py"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load trained joint-constraint checkpoints and evaluate zero-shot external generalization on ds007216."
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config file; keys match CLI argument names.")
    parser.add_argument("--train-output-root", type=Path, default=DEFAULT_TRAIN_OUTPUT_ROOT)
    parser.add_argument("--scoring-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--participants-tsv", type=Path, default=DEFAULT_PARTICIPANTS_TSV)
    parser.add_argument("--ds007216-script", type=Path, default=DEFAULT_TEST_TOOL)
    parser.add_argument(
        "--label-mode",
        choices=["stai_state_mean", "stai_v1_score", "stai_v2_score", "gad7_score"],
        default="stai_state_mean",
        help="Which cleaned ds007216 label to use for external evaluation.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["shared_only", "mean_adapter", "proxy_dataset"],
        default=["shared_only", "mean_adapter"],
        help="External inference strategies for unseen-dataset adapter handling.",
    )
    parser.add_argument(
        "--proxy-dataset",
        choices=["original_local", "ds003478", "ds007609"],
        default="ds007609",
        help="Used only when 'proxy_dataset' strategy is enabled.",
    )
    parser.add_argument(
        "--binary-mode",
        choices=["median", "gad7_ge10", "gad7_official_any", "gad7_official_strict"],
        default="median",
        help=(
            "How to convert the external score into a binary label. "
            "'median' keeps the previous median split; "
            "'gad7_ge10' uses GAD-7 <10 vs >=10; "
            "'gad7_official_any' uses GAD-7 0-4 vs >=5; "
            "'gad7_official_strict' uses GAD-7 0-4 vs >=10 and excludes 5-9."
        ),
    )
    parser.add_argument("--gray-z", type=float, default=0.35)
    parser.add_argument("--rerun-scoring", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return apply_json_config(parser.parse_args())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: object) -> float:
    out = float(str(value).strip())
    if not math.isfinite(out):
        raise ValueError(f"Non-finite float value: {value!r}")
    return out


def valid_scale_value(value: object, missing_floor: float = -8.5) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= missing_floor:
        return None
    return float(out)


def canonical_subject(subject: object) -> str | None:
    if subject is None:
        return None
    text = str(subject).strip().lower()
    for prefix in ("sub-", "subject-", "participant-"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def prepare_clean_label_file(label_mode: str, participants_tsv: Path, out_path: Path) -> tuple[Path, str, dict]:
    rows = read_tsv(participants_tsv)
    clean_rows = []
    subject_scores = []

    for row in rows:
        participant_id = str(row["participant_id"]).strip()
        stai_v1 = valid_scale_value(row.get("stai_v1_score"))
        stai_v2 = valid_scale_value(row.get("stai_v2_score"))
        gad7 = valid_scale_value(row.get("gad7_score"))

        if label_mode == "stai_state_mean":
            values = [value for value in (stai_v1, stai_v2) if value is not None]
            score = None if not values else float(np.mean(values))
        elif label_mode == "stai_v1_score":
            score = stai_v1
        elif label_mode == "stai_v2_score":
            score = stai_v2
        elif label_mode == "gad7_score":
            score = gad7
        else:
            raise ValueError(f"Unsupported label_mode={label_mode!r}")

        if score is None:
            continue

        subject_scores.append(float(score))
        clean_rows.append(
            {
                "participant_id": participant_id,
                label_mode: float(score),
                "stai_v1_score_clean": "" if stai_v1 is None else float(stai_v1),
                "stai_v2_score_clean": "" if stai_v2 is None else float(stai_v2),
                "gad7_score_clean": "" if gad7 is None else float(gad7),
                "n_valid_stai": int(sum(value is not None for value in (stai_v1, stai_v2))),
            }
        )

    if len(clean_rows) < 4:
        raise RuntimeError(f"Too few valid subjects after ds007216 label cleanup: n={len(clean_rows)}")

    write_csv(out_path.with_suffix(".csv"), clean_rows)
    tsv_rows = [{key: str(value) for key, value in row.items()} for row in clean_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(tsv_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(tsv_rows)

    summary = {
        "participants_tsv": str(participants_tsv),
        "label_mode": label_mode,
        "n_subjects": int(len(clean_rows)),
        "score_min": float(np.min(subject_scores)),
        "score_median": float(np.median(subject_scores)),
        "score_max": float(np.max(subject_scores)),
        "score_mean": float(np.mean(subject_scores)),
        "score_std": float(np.std(subject_scores)),
    }
    write_json(out_path.with_suffix(".json"), summary)
    return out_path, label_mode, summary


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


def ensure_ds007216_scoring(args: argparse.Namespace, clean_label_file: Path, clean_label_column: str) -> Path:
    subject_features_path = args.scoring_dir / "subject_features.csv"
    if args.skip_scoring and not subject_features_path.exists():
        raise FileNotFoundError(f"--skip-scoring was set but {subject_features_path} does not exist")
    if subject_features_path.exists() and not args.rerun_scoring:
        return subject_features_path
    raise FileNotFoundError(
        "未找到 ds007216 subject_features.csv。请先运行 "
        "`python -m anxiety_eeg.features.score_ds007216 --output-dir features/external/ds007216` "
        "生成特征，再使用 `--skip-scoring` 调用本外部评估脚本。"
    )

    args.scoring_dir.mkdir(parents=True, exist_ok=True)
    script_path = args.ds007216_script.resolve()
    specialized_extractor = (CURRENT_DIR / "extract_ds007216_single_run.py").resolve()
    if script_path == specialized_extractor:
        command = [
            sys.executable,
            str(args.ds007216_script),
            "--output-dir",
            str(args.scoring_dir),
            "--participants-tsv",
            str(args.participants_tsv),
            "--label-file",
            str(clean_label_file),
            "--label-mode",
            str(args.label_mode),
        ]
    else:
        command = [
            sys.executable,
            str(args.ds007216_script),
            "--output-dir",
            str(args.scoring_dir),
            "--label-file",
            str(clean_label_file),
            "--label-column",
            str(clean_label_column),
        ]
    run_logged_subprocess(
        command=command,
        cwd=REPO_ROOT,
        log_path=args.scoring_dir / "extract_ds007216.log",
    )
    if not subject_features_path.exists():
        raise FileNotFoundError(f"Expected subject_features.csv after ds007216 extraction: {subject_features_path}")
    return subject_features_path


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
    payload = {
        "strategies": list(strategies),
        "by_strategy": {},
    }
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


def read_external_subject_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No rows found in external subject feature csv: {path}")
    return rows


def build_external_binary_view(
    rows: list[dict],
    label_mode: str,
    binary_mode: str,
    gray_z: float,
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    scores = np.array([finite_float(row["anxiety"]) for row in rows], dtype=np.float32)
    if binary_mode == "median":
        center = float(np.median(scores))
        std = float(np.std(scores))
        if not math.isfinite(std) or std < EPS:
            std = 1.0
        labels = (scores >= center).astype(np.int64)
        in_gray_zone = np.logical_and(
            scores > float(center - float(gray_z) * std),
            scores < float(center + float(gray_z) * std),
        )
        thresholds = {
            "binary_mode": str(binary_mode),
            "label_definition": "low = score < median, high = score >= median",
            "center": center,
            "std": std,
            "gray_z": float(gray_z),
            "low_threshold": float(center - float(gray_z) * std),
            "high_threshold": float(center + float(gray_z) * std),
            "original_subjects": int(len(rows)),
            "kept_subjects": int(len(rows)),
            "excluded_subjects": 0,
        }
        eval_rows = list(rows)
    else:
        if str(label_mode) != "gad7_score":
            raise ValueError(
                f"binary_mode={binary_mode!r} requires --label-mode gad7_score, got {label_mode!r}"
            )

        score_std = float(np.std(scores))
        if not math.isfinite(score_std) or score_std < EPS:
            score_std = 1.0

        if binary_mode == "gad7_ge10":
            keep_mask = np.ones(scores.shape[0], dtype=bool)
            labels = (scores >= 10.0).astype(np.int64)
            in_gray_zone = np.zeros(scores.shape[0], dtype=bool)
            thresholds = {
                "binary_mode": str(binary_mode),
                "label_definition": "low = GAD-7 < 10, high = GAD-7 >= 10",
                "center": 10.0,
                "std": score_std,
                "gray_z": 0.0,
                "low_threshold": 10.0,
                "high_threshold": 10.0,
                "original_subjects": int(len(rows)),
                "kept_subjects": int(len(rows)),
                "excluded_subjects": 0,
            }
        elif binary_mode == "gad7_official_any":
            keep_mask = np.ones(scores.shape[0], dtype=bool)
            labels = (scores >= 5.0).astype(np.int64)
            in_gray_zone = np.zeros(scores.shape[0], dtype=bool)
            thresholds = {
                "binary_mode": str(binary_mode),
                "label_definition": "low = GAD-7 0-4, high = GAD-7 >= 5",
                "center": 5.0,
                "std": score_std,
                "gray_z": 0.0,
                "low_threshold": 4.0,
                "high_threshold": 5.0,
                "original_subjects": int(len(rows)),
                "kept_subjects": int(len(rows)),
                "excluded_subjects": 0,
            }
        elif binary_mode == "gad7_official_strict":
            keep_mask = np.logical_or(scores <= 4.0, scores >= 10.0)
            labels = (scores[keep_mask] >= 10.0).astype(np.int64)
            in_gray_zone = np.zeros(np.sum(keep_mask), dtype=bool)
            thresholds = {
                "binary_mode": str(binary_mode),
                "label_definition": "low = GAD-7 0-4, high = GAD-7 >= 10, excluded = 5-9",
                "center": 10.0,
                "std": float(np.std(scores[keep_mask])) if np.any(keep_mask) else score_std,
                "gray_z": 0.0,
                "low_threshold": 4.0,
                "high_threshold": 10.0,
                "original_subjects": int(len(rows)),
                "kept_subjects": int(np.sum(keep_mask)),
                "excluded_subjects": int(np.sum(~keep_mask)),
            }
        else:
            raise ValueError(f"Unsupported binary_mode={binary_mode!r}")

        eval_rows = [row for row, keep in zip(rows, keep_mask) if keep]

    if not eval_rows:
        raise RuntimeError(f"binary_mode={binary_mode!r} removed every external subject.")

    n_low = int(np.sum(labels == 0))
    n_high = int(np.sum(labels == 1))
    thresholds["n_low"] = n_low
    thresholds["n_high"] = n_high
    if n_low == 0 or n_high == 0:
        raise RuntimeError(
            f"binary_mode={binary_mode!r} produced a single class on ds007216: "
            f"n_low={n_low}, n_high={n_high}, kept_subjects={len(eval_rows)}"
        )

    return eval_rows, labels, in_gray_zone, thresholds


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


def build_external_matrix(rows: list[dict], feature_names: list[str], means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    out = []
    for row in rows:
        values = []
        for feature_name in feature_names:
            if feature_name not in row:
                raise KeyError(f"External row missing required feature {feature_name!r}")
            raw = finite_float(row[feature_name])
            values.append(float((raw - means[feature_name]) / (stds[feature_name] + EPS)))
        out.append(values)
    return np.asarray(out, dtype=np.float32)


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


def adapter_terms(
    model: JointConstraintNet,
    strategy: str,
    dataset_order: list[str],
    proxy_dataset: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_dim = int(model.classifier.net[0].in_features)
    if strategy == "shared_only":
        return torch.zeros(hidden_dim, device=device), torch.tensor(0.0, device=device)

    if strategy == "mean_adapter":
        embedding = model.dataset_embedding.weight.detach().mean(dim=0, keepdim=True)
        adapter_hidden = model.adapter(embedding).squeeze(0)
        bias = model.dataset_bias.weight.detach().mean()
        return adapter_hidden.to(device), bias.to(device)

    if strategy == "proxy_dataset":
        if proxy_dataset not in dataset_order:
            raise KeyError(f"proxy_dataset={proxy_dataset!r} not in dataset_order={dataset_order}")
        proxy_index = int(dataset_order.index(proxy_dataset))
        embedding = model.dataset_embedding.weight.detach()[proxy_index : proxy_index + 1]
        adapter_hidden = model.adapter(embedding).squeeze(0)
        bias = model.dataset_bias.weight.detach()[proxy_index, 0]
        return adapter_hidden.to(device), bias.to(device)

    raise ValueError(f"Unsupported strategy={strategy!r}")


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
    adapter_hidden, bias = adapter_terms(model, strategy, dataset_order, proxy_dataset, device)
    adapted = shared + adapter_hidden.unsqueeze(0)
    logits = model.classifier(adapted).squeeze(-1) + bias
    probs = torch.sigmoid(logits)
    return logits.detach().cpu().numpy(), probs.detach().cpu().numpy()


def evaluate_one_seed(
    seed_dir: Path,
    external_rows: list[dict],
    labels: np.ndarray,
    in_gray_zone: np.ndarray,
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
    x = build_external_matrix(external_rows, list(config["input_features"]), means, stds)

    anxiety = np.asarray([finite_float(row["anxiety"]) for row in external_rows], dtype=np.float32)

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
                "binary_mode": str(thresholds["binary_mode"]),
                "center": float(thresholds["center"]),
                "std": float(thresholds["std"]),
                "low_threshold": float(thresholds["low_threshold"]),
                "high_threshold": float(thresholds["high_threshold"]),
                "excluded_subjects": int(thresholds.get("excluded_subjects", 0)),
            }
        )
        prediction_rows_by_strategy[strategy] = [
            {
                "seed": int(config["seed"]),
                "strategy": strategy,
                "subject": str(row["subject"]),
                "score_name": str(row.get("score_name", "")),
                "context": str(row.get("context", "")),
                "anxiety": float(row["anxiety"]),
                "label": int(label),
                "in_gray_zone": bool(gray),
                "logit": float(logit),
                "prob": float(prob),
                "pred_label": int(prob >= 0.5),
            }
            for row, label, gray, logit, prob in zip(external_rows, labels, in_gray_zone, logits, probs)
        ]

    run_meta = {
        "seed": int(config["seed"]),
        "checkpoint": str(checkpoint_path),
        "input_features": list(config["input_features"]),
        "dataset_order": list(config["dataset_order"]),
        "hidden_dim": int(config["hidden_dim"]),
        "adapter_dim": int(config["adapter_dim"]),
        "dropout": float(config["dropout"]),
    }
    return summary_rows, prediction_rows_by_strategy, run_meta


def build_markdown_summary(payload: dict, rows: list[dict], config: dict) -> str:
    lines = [
        "# ds007216 External Generalization Summary",
        "",
        f"- Train root: `{config['train_output_root']}`",
        f"- External scoring dir: `{config['scoring_dir']}`",
        f"- Label mode: `{config['label_mode']}`",
        f"- Label file: `{config['label_file']}`",
        f"- Binary mode: `{config['binary_mode']}`",
        f"- Label definition: `{config['label_definition']}`",
        f"- External subjects kept: `{config['kept_subjects']}` / `{config['original_subjects']}`",
        f"- Proxy dataset: `{config['proxy_dataset']}`",
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
            "| Seed | Strategy | BalAcc | AUC | Ext BalAcc | Ext AUC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["strategy"]), int(item["seed"]))):
        lines.append(
            f"| {row['seed']} | {row['strategy']} | "
            f"{float(row['balanced_accuracy']):.3f} | {float(row['roc_auc']):.3f} | "
            f"{float(row['extreme_balanced_accuracy']):.3f} | {float(row['extreme_roc_auc']):.3f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    clean_label_file, clean_label_column, label_summary = prepare_clean_label_file(
        label_mode=args.label_mode,
        participants_tsv=args.participants_tsv,
        out_path=args.label_file,
    )
    subject_features_path = ensure_ds007216_scoring(
        args=args,
        clean_label_file=clean_label_file,
        clean_label_column=clean_label_column,
    )
    external_rows_all = read_external_subject_rows(subject_features_path)
    external_rows, external_labels, external_gray, thresholds = build_external_binary_view(
        rows=external_rows_all,
        label_mode=args.label_mode,
        binary_mode=args.binary_mode,
        gray_z=args.gray_z,
    )

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
            labels=external_labels,
            in_gray_zone=external_gray,
            thresholds=thresholds,
            strategies=list(args.strategies),
            proxy_dataset=args.proxy_dataset,
            device=device,
        )
        summary_rows_all.extend(summary_rows)
        run_meta_rows.append(run_meta)
        for strategy, prediction_rows in prediction_rows_by_strategy.items():
            write_csv(
                prediction_root / f"{seed_dir.name}_{strategy}_predictions_ds007216.csv",
                prediction_rows,
            )

    aggregate_rows, payload = aggregate_payload(summary_rows_all, list(args.strategies))
    runtime_sec = float(time.time() - started)
    payload["runtime_sec"] = runtime_sec
    payload["n_seeds"] = int(len(seed_dirs))
    payload["seed_values"] = [int(path.name.split("_", 1)[1]) for path in seed_dirs]
    payload["external_thresholds"] = thresholds
    payload["label_summary"] = label_summary
    payload["train_output_root"] = str(args.train_output_root.resolve())
    payload["scoring_dir"] = str(args.scoring_dir.resolve())
    payload["label_file"] = str(clean_label_file.resolve())
    payload["label_mode"] = str(args.label_mode)
    payload["binary_mode"] = str(args.binary_mode)
    payload["proxy_dataset"] = str(args.proxy_dataset)

    write_csv(args.results_dir / "summary_all_seeds_ds007216.csv", summary_rows_all)
    write_csv(args.results_dir / "summary_aggregate_ds007216.csv", aggregate_rows)
    write_json(args.results_dir / "summary_aggregate_ds007216.json", payload)
    write_json(args.results_dir / "run_meta_ds007216.json", {"rows": run_meta_rows})
    write_text(
        args.results_dir / "summary_ds007216.md",
        build_markdown_summary(
            payload=payload,
            rows=summary_rows_all,
            config={
                "train_output_root": str(args.train_output_root),
                "scoring_dir": str(args.scoring_dir),
                "label_mode": args.label_mode,
                "label_file": str(clean_label_file),
                "binary_mode": args.binary_mode,
                "label_definition": str(thresholds["label_definition"]),
                "original_subjects": int(thresholds["original_subjects"]),
                "kept_subjects": int(thresholds["kept_subjects"]),
                "proxy_dataset": args.proxy_dataset,
            },
        ),
    )

    print(f"[Device] {device}")
    print(f"[ds007216] subject_features={subject_features_path}")
    print(f"[ds007216] results_dir={args.results_dir.resolve()}")
    print(
        f"[ds007216] binary_mode={args.binary_mode} "
        f"kept={thresholds['kept_subjects']}/{thresholds['original_subjects']} "
        f"n_low={thresholds['n_low']} n_high={thresholds['n_high']}"
    )
    for strategy in args.strategies:
        metrics = payload["by_strategy"].get(strategy, {})
        bal = metrics.get("balanced_accuracy")
        auc = metrics.get("roc_auc")
        ext_bal = metrics.get("extreme_balanced_accuracy")
        ext_auc = metrics.get("extreme_roc_auc")
        if not bal or not auc:
            continue
        print(
            f"[Strategy:{strategy}] "
            f"bal={bal['mean']:.4f}+/-{bal['std']:.4f} "
            f"auc={auc['mean']:.4f}+/-{auc['std']:.4f} "
            f"ext_bal={ext_bal['mean']:.4f}+/-{ext_bal['std']:.4f} "
            f"ext_auc={ext_auc['mean']:.4f}+/-{ext_auc['std']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
