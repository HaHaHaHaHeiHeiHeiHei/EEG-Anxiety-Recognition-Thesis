"""中文说明

用途：
    分析 ds007216 外部预测是否呈现系统性反向判别，用于判断域偏移方向问题。
输入：
    `outputs/external_ds007216/predictions` 中的 seed-level 预测 CSV。
输出：
    反向 AUC、原始 AUC、相关系数和 Markdown 报告。
快速运行：
    `python -m anxiety_eeg.external.analyze_reverse_discrimination_ds007216 --predictions-dir outputs/external_ds007216/predictions`
论文对应：
    第 5 章 ds007216 reverse-discrimination / direction mismatch 分析。
注意事项：
    反向判别只用于解释外部失败模式，不应当写成模型可直接上线应用。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "external_ds007216"
DEFAULT_PREDICTIONS_DIR = DEFAULT_RESULTS_DIR / "predictions"
DEFAULT_PER_SEED_CSV = DEFAULT_RESULTS_DIR / "reverse_discrimination_all_seeds_ds007216.csv"
DEFAULT_AGG_CSV = DEFAULT_RESULTS_DIR / "reverse_discrimination_aggregate_ds007216.csv"
DEFAULT_AGG_JSON = DEFAULT_RESULTS_DIR / "reverse_discrimination_aggregate_ds007216.json"
DEFAULT_MD = DEFAULT_RESULTS_DIR / "reverse_discrimination_ds007216.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether ds007216 external results indicate systematic sign inversion / reverse discrimination."
    )
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--per-seed-csv", type=Path, default=DEFAULT_PER_SEED_CSV)
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGG_CSV)
    parser.add_argument("--aggregate-json", type=Path, default=DEFAULT_AGG_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        return float("nan")


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    if np.unique(np.round(x, 12)).size < 2 or np.unique(np.round(y, 12)).size < 2:
        return float("nan")
    try:
        if method == "pearson":
            return float(pearsonr(x, y).statistic)
        return float(spearmanr(x, y).statistic)
    except Exception:
        return float("nan")


def mean_std_text(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def analyze_one_prediction_file(path: Path) -> dict:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")

    seed = int(rows[0]["seed"])
    strategy = str(rows[0]["strategy"])
    anxiety = np.asarray([float(row["anxiety"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    in_gray_zone = np.asarray(
        [str(row["in_gray_zone"]).strip().lower() == "true" for row in rows],
        dtype=bool,
    )
    probs = np.asarray([float(row["prob"]) for row in rows], dtype=np.float64)
    preds = np.asarray([int(row["pred_label"]) for row in rows], dtype=np.int64)

    extreme_mask = ~in_gray_zone
    flipped_probs = 1.0 - probs
    flipped_preds = 1 - preds

    out = {
        "seed": seed,
        "strategy": strategy,
        "n_subjects": int(len(rows)),
        "gray_subjects": int(np.sum(in_gray_zone)),
        "orig_balanced_accuracy": safe_balanced_accuracy(labels, preds),
        "orig_roc_auc": safe_roc_auc(labels, probs),
        "flip_balanced_accuracy": safe_balanced_accuracy(labels, flipped_preds),
        "flip_roc_auc": safe_roc_auc(labels, flipped_probs),
        "prob_anxiety_pearson": safe_corr(probs, anxiety, method="pearson"),
        "prob_anxiety_spearman": safe_corr(probs, anxiety, method="spearman"),
        "flipprob_anxiety_pearson": safe_corr(flipped_probs, anxiety, method="pearson"),
        "flipprob_anxiety_spearman": safe_corr(flipped_probs, anxiety, method="spearman"),
    }

    if np.sum(extreme_mask) >= 3 and len(np.unique(labels[extreme_mask])) > 1:
        out["orig_extreme_balanced_accuracy"] = safe_balanced_accuracy(labels[extreme_mask], preds[extreme_mask])
        out["orig_extreme_roc_auc"] = safe_roc_auc(labels[extreme_mask], probs[extreme_mask])
        out["flip_extreme_balanced_accuracy"] = safe_balanced_accuracy(labels[extreme_mask], flipped_preds[extreme_mask])
        out["flip_extreme_roc_auc"] = safe_roc_auc(labels[extreme_mask], flipped_probs[extreme_mask])
        out["extreme_prob_anxiety_pearson"] = safe_corr(probs[extreme_mask], anxiety[extreme_mask], method="pearson")
        out["extreme_prob_anxiety_spearman"] = safe_corr(
            probs[extreme_mask], anxiety[extreme_mask], method="spearman"
        )
        out["extreme_flipprob_anxiety_pearson"] = safe_corr(
            flipped_probs[extreme_mask], anxiety[extreme_mask], method="pearson"
        )
        out["extreme_flipprob_anxiety_spearman"] = safe_corr(
            flipped_probs[extreme_mask], anxiety[extreme_mask], method="spearman"
        )
    else:
        out["orig_extreme_balanced_accuracy"] = float("nan")
        out["orig_extreme_roc_auc"] = float("nan")
        out["flip_extreme_balanced_accuracy"] = float("nan")
        out["flip_extreme_roc_auc"] = float("nan")
        out["extreme_prob_anxiety_pearson"] = float("nan")
        out["extreme_prob_anxiety_spearman"] = float("nan")
        out["extreme_flipprob_anxiety_pearson"] = float("nan")
        out["extreme_flipprob_anxiety_spearman"] = float("nan")

    out["auc_gain_after_flip"] = float(out["flip_roc_auc"] - out["orig_roc_auc"])
    out["bal_gain_after_flip"] = float(out["flip_balanced_accuracy"] - out["orig_balanced_accuracy"])
    out["extreme_auc_gain_after_flip"] = float(out["flip_extreme_roc_auc"] - out["orig_extreme_roc_auc"])
    out["extreme_bal_gain_after_flip"] = float(
        out["flip_extreme_balanced_accuracy"] - out["orig_extreme_balanced_accuracy"]
    )
    out["seed_supports_flip"] = int(
        math.isfinite(out["orig_roc_auc"])
        and math.isfinite(out["flip_roc_auc"])
        and out["orig_roc_auc"] < 0.5
        and out["flip_roc_auc"] > 0.5
        and math.isfinite(out["prob_anxiety_spearman"])
        and out["prob_anxiety_spearman"] < 0.0
    )
    return out


def finite_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def aggregate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)

    aggregate_csv_rows = []
    aggregate_payload = {"by_strategy": {}}
    metric_names = [
        "orig_balanced_accuracy",
        "orig_roc_auc",
        "flip_balanced_accuracy",
        "flip_roc_auc",
        "orig_extreme_balanced_accuracy",
        "orig_extreme_roc_auc",
        "flip_extreme_balanced_accuracy",
        "flip_extreme_roc_auc",
        "prob_anxiety_pearson",
        "prob_anxiety_spearman",
        "flipprob_anxiety_pearson",
        "flipprob_anxiety_spearman",
        "extreme_prob_anxiety_pearson",
        "extreme_prob_anxiety_spearman",
        "extreme_flipprob_anxiety_pearson",
        "extreme_flipprob_anxiety_spearman",
        "auc_gain_after_flip",
        "bal_gain_after_flip",
        "extreme_auc_gain_after_flip",
        "extreme_bal_gain_after_flip",
    ]

    for strategy, strategy_rows in sorted(grouped.items()):
        payload = {
            "n_seeds": int(len(strategy_rows)),
            "n_auc_below_0_5": int(sum(row["orig_roc_auc"] < 0.5 for row in strategy_rows if math.isfinite(row["orig_roc_auc"]))),
            "n_flip_auc_above_0_5": int(
                sum(row["flip_roc_auc"] > 0.5 for row in strategy_rows if math.isfinite(row["flip_roc_auc"]))
            ),
            "n_negative_spearman": int(
                sum(row["prob_anxiety_spearman"] < 0.0 for row in strategy_rows if math.isfinite(row["prob_anxiety_spearman"]))
            ),
            "n_seed_supports_flip": int(sum(int(row["seed_supports_flip"]) for row in strategy_rows)),
        }
        for metric_name in metric_names:
            mean, std = finite_mean_std([float(row[metric_name]) for row in strategy_rows])
            payload[metric_name] = {"mean": mean, "std": std}
            aggregate_csv_rows.append(
                {
                    "strategy": strategy,
                    "metric": metric_name,
                    "mean": mean,
                    "std": std,
                }
            )

        payload["systematic_sign_flip_likely"] = bool(
            payload["n_seed_supports_flip"] == payload["n_seeds"] and payload["n_seeds"] > 0
        )
        aggregate_payload["by_strategy"][strategy] = payload

    return aggregate_csv_rows, aggregate_payload


def build_markdown(rows: list[dict], payload: dict) -> str:
    lines = [
        "# ds007216 Reverse Discrimination Analysis",
        "",
        "## Conclusion",
        "",
    ]
    for strategy, info in sorted(payload["by_strategy"].items()):
        conclusion = "Yes" if info["systematic_sign_flip_likely"] else "Partially / No"
        lines.append(
            f"- `{strategy}`: systematic sign flip likely = `{conclusion}`; "
            f"`AUC<0.5` seeds = `{info['n_auc_below_0_5']}/{info['n_seeds']}`, "
            f"`negative Spearman` seeds = `{info['n_negative_spearman']}/{info['n_seeds']}`, "
            f"`flip-supported` seeds = `{info['n_seed_supports_flip']}/{info['n_seeds']}`."
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Strategy | Orig AUC | Flipped AUC | Orig BalAcc | Flipped BalAcc | Prob~Anxiety Spearman | Flipped Prob~Anxiety Spearman |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for strategy, info in sorted(payload["by_strategy"].items()):
        lines.append(
            "| "
            + strategy
            + " | "
            + mean_std_text(info["orig_roc_auc"]["mean"], info["orig_roc_auc"]["std"])
            + " | "
            + mean_std_text(info["flip_roc_auc"]["mean"], info["flip_roc_auc"]["std"])
            + " | "
            + mean_std_text(info["orig_balanced_accuracy"]["mean"], info["orig_balanced_accuracy"]["std"])
            + " | "
            + mean_std_text(info["flip_balanced_accuracy"]["mean"], info["flip_balanced_accuracy"]["std"])
            + " | "
            + mean_std_text(info["prob_anxiety_spearman"]["mean"], info["prob_anxiety_spearman"]["std"])
            + " | "
            + mean_std_text(
                info["flipprob_anxiety_spearman"]["mean"],
                info["flipprob_anxiety_spearman"]["std"],
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Seed Table",
            "",
            "| Seed | Strategy | Orig AUC | Flipped AUC | Orig BalAcc | Flipped BalAcc | Spearman(prob, anxiety) | Supports Flip |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["strategy"]), int(item["seed"]))):
        lines.append(
            f"| {row['seed']} | {row['strategy']} | "
            f"{float(row['orig_roc_auc']):.3f} | {float(row['flip_roc_auc']):.3f} | "
            f"{float(row['orig_balanced_accuracy']):.3f} | {float(row['flip_balanced_accuracy']):.3f} | "
            f"{float(row['prob_anxiety_spearman']):.3f} | {int(row['seed_supports_flip'])} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    prediction_files = sorted(args.predictions_dir.glob("seed_*_predictions_ds007216.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {args.predictions_dir}")

    per_seed_rows = [analyze_one_prediction_file(path) for path in prediction_files]
    aggregate_csv_rows, aggregate_payload = aggregate_rows(per_seed_rows)

    write_csv(args.per_seed_csv, per_seed_rows)
    write_csv(args.aggregate_csv, aggregate_csv_rows)
    write_json(args.aggregate_json, aggregate_payload)
    write_text(args.markdown, build_markdown(per_seed_rows, aggregate_payload))

    for strategy, info in sorted(aggregate_payload["by_strategy"].items()):
        print(
            f"[Reverse:{strategy}] "
            f"orig_auc={info['orig_roc_auc']['mean']:.4f}+/-{info['orig_roc_auc']['std']:.4f} "
            f"flip_auc={info['flip_roc_auc']['mean']:.4f}+/-{info['flip_roc_auc']['std']:.4f} "
            f"orig_bal={info['orig_balanced_accuracy']['mean']:.4f}+/-{info['orig_balanced_accuracy']['std']:.4f} "
            f"flip_bal={info['flip_balanced_accuracy']['mean']:.4f}+/-{info['flip_balanced_accuracy']['std']:.4f} "
            f"spearman={info['prob_anxiety_spearman']['mean']:.4f}+/-{info['prob_anxiety_spearman']['std']:.4f} "
            f"flip_supported={info['n_seed_supports_flip']}/{info['n_seeds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
