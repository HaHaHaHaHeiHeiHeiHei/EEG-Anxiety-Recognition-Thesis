"""中文说明

用途：
    分析内部数据集、Mendeley 和 ds007216 上共享 EEG 频谱特征的方向一致性，
    并汇总外部预测文件的 AUC 显著性/反向判别证据。
输入：
    内部 `--features-root`，外部 `--mendeley-features`、`--ds007216-features`，
    以及可选预测目录。
输出：
    feature direction CSV、heatmap、外部 AUC 检验表和中文/英文混合分析报告。
快速运行：
    `python -m anxiety_eeg.analysis.analyze_direction_consistency --features-root features/subject_features --skip-prediction-tests`
论文对应：
    第 5 章“特征方向与跨 seed 稳定性”和外部兼容性审计。
注意事项：
    ds007216 反向结果用于说明 domain-direction mismatch，不应被写成最终泛化性能。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


matplotlib.use("Agg")

from anxiety_eeg.config import apply_json_config
from anxiety_eeg.data.joint_dataset import (  # noqa: E402
    DEFAULT_FEATURES_ROOT,
    SHARED_COMMON_INPUT_FEATURES,
    dataset_csvs_from_root,
    read_subject_feature_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "direction_consistency"
DEFAULT_MENDELEY_CSV = REPO_ROOT / "features" / "external" / "mendeley" / "subject_features.csv"
DEFAULT_DS007216_CSV = REPO_ROOT / "features" / "external" / "ds007216" / "subject_features.csv"
DEFAULT_MENDELEY_PRED_DIR = REPO_ROOT / "outputs" / "external_mendeley" / "predictions"
DEFAULT_DS007216_PRED_DIR = REPO_ROOT / "outputs" / "external_ds007216" / "predictions"
DEFAULT_DS007216_REVERSE_CSV = REPO_ROOT / "outputs" / "external_ds007216" / "reverse_discrimination_all_seeds_ds007216.csv"

CORE_FEATURES = list(SHARED_COMMON_INPUT_FEATURES)
HEATMAP_DATASETS = ["original_local", "ds003478", "ds007609", "mendeley", "ds007216"]
REFERENCE_DATASETS = ["original_local", "ds003478", "ds007609", "mendeley"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-dataset feature directions and external significance."
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config file; keys match CLI argument names.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--mendeley-features", type=Path, default=DEFAULT_MENDELEY_CSV)
    parser.add_argument("--ds007216-features", type=Path, default=DEFAULT_DS007216_CSV)
    parser.add_argument("--mendeley-pred-dir", type=Path, default=DEFAULT_MENDELEY_PRED_DIR)
    parser.add_argument("--ds007216-pred-dir", type=Path, default=DEFAULT_DS007216_PRED_DIR)
    parser.add_argument("--ds007216-reverse-csv", type=Path, default=DEFAULT_DS007216_REVERSE_CSV)
    parser.add_argument("--skip-prediction-tests", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--permutation-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260425)
    return apply_json_config(parser.parse_args())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sign_label(value: float, tol: float = 1e-8) -> str:
    if not math.isfinite(float(value)):
        return "0"
    if value > tol:
        return "+"
    if value < -tol:
        return "-"
    return "0"


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    try:
        result = fn(x, y)
    except Exception:
        return float("nan"), float("nan")
    statistic = getattr(result, "statistic", result[0] if isinstance(result, tuple) else float("nan"))
    pvalue = getattr(result, "pvalue", result[1] if isinstance(result, tuple) else float("nan"))
    return float(statistic), float(pvalue)


def cohens_d(high_values: np.ndarray, low_values: np.ndarray) -> float:
    if len(high_values) < 2 or len(low_values) < 2:
        return float("nan")
    high_mean = float(np.mean(high_values))
    low_mean = float(np.mean(low_values))
    high_var = float(np.var(high_values, ddof=1))
    low_var = float(np.var(low_values, ddof=1))
    pooled_num = (len(high_values) - 1) * high_var + (len(low_values) - 1) * low_var
    pooled_den = len(high_values) + len(low_values) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_std = math.sqrt(max(pooled_num / pooled_den, 1e-12))
    return (high_mean - low_mean) / pooled_std


def dataset_binary_groups(rows: list) -> tuple[np.ndarray, np.ndarray]:
    anxiety = np.asarray([float(row.anxiety) for row in rows], dtype=np.float64)
    unique = np.unique(anxiety)
    if unique.size == 2 and set(unique.tolist()).issubset({0.0, 1.0}):
        low_mask = anxiety < 0.5
        high_mask = anxiety >= 0.5
    else:
        median = float(np.median(anxiety))
        low_mask = anxiety < median
        high_mask = anxiety >= median
    return low_mask, high_mask


def build_direction_rows(dataset_name: str, csv_path: Path, features: list[str]) -> list[dict]:
    rows = read_subject_feature_csv(csv_path, dataset_name=dataset_name)
    anxiety = np.asarray([float(row.anxiety) for row in rows], dtype=np.float64)
    low_mask, high_mask = dataset_binary_groups(rows)

    out_rows: list[dict] = []
    for feature in features:
        values = np.asarray([float(row.features[feature]) for row in rows], dtype=np.float64)
        spearman_rho, spearman_p = safe_corr(stats.spearmanr, values, anxiety)
        pearson_r, pearson_p = safe_corr(stats.pearsonr, values, anxiety)

        low_values = values[low_mask]
        high_values = values[high_mask]
        diff_high_minus_low = float(np.mean(high_values) - np.mean(low_values))
        effect_d = float(cohens_d(high_values, low_values))

        out_rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "n_subjects": int(len(rows)),
                "low_n": int(np.sum(low_mask)),
                "high_n": int(np.sum(high_mask)),
                "score_mean": float(np.mean(anxiety)),
                "score_std": float(np.std(anxiety, ddof=0)),
                "feature_mean": float(np.mean(values)),
                "feature_std": float(np.std(values, ddof=0)),
                "low_mean": float(np.mean(low_values)),
                "high_mean": float(np.mean(high_values)),
                "diff_high_minus_low": diff_high_minus_low,
                "effect_size_d": effect_d,
                "direction_by_diff": sign_label(diff_high_minus_low),
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "direction_by_rho": sign_label(spearman_rho),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
            }
        )
    return out_rows


def summarize_consistency(direction_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature, feature_df in direction_df.groupby("feature"):
        ref_df = feature_df[feature_df["dataset"].isin(REFERENCE_DATASETS)]
        ref_signs = [sign for sign in ref_df["direction_by_diff"].tolist() if sign in {"+", "-"}]
        pos_count = int(sum(sign == "+" for sign in ref_signs))
        neg_count = int(sum(sign == "-" for sign in ref_signs))
        if pos_count == neg_count:
            reference_sign = "0"
        else:
            reference_sign = "+" if pos_count > neg_count else "-"

        ds007216_sign = str(
            feature_df.loc[feature_df["dataset"] == "ds007216", "direction_by_diff"].iloc[0]
        )
        mendeley_sign = str(
            feature_df.loc[feature_df["dataset"] == "mendeley", "direction_by_diff"].iloc[0]
        )
        supporting = int(sum(sign == reference_sign for sign in ref_signs)) if reference_sign in {"+", "-"} else 0
        opposing = int(sum(sign != reference_sign for sign in ref_signs)) if reference_sign in {"+", "-"} else 0
        rows.append(
            {
                "feature": feature,
                "reference_direction": reference_sign,
                "reference_positive_datasets": pos_count,
                "reference_negative_datasets": neg_count,
                "reference_supporting_datasets": supporting,
                "reference_opposing_datasets": opposing,
                "mendeley_direction": mendeley_sign,
                "ds007216_direction": ds007216_sign,
                "ds007216_matches_reference": int(reference_sign in {"+", "-"} and ds007216_sign == reference_sign),
                "ds007216_reverses_reference": int(
                    reference_sign in {"+", "-"}
                    and ds007216_sign in {"+", "-"}
                    and ds007216_sign != reference_sign
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def make_heatmap(direction_df: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        direction_df.assign(effect_value=direction_df["effect_size_d"].astype(float))
        .pivot(index="feature", columns="dataset", values="effect_value")
        .reindex(index=CORE_FEATURES, columns=HEATMAP_DATASETS)
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("High-minus-low feature direction across datasets")
    for row_idx, feature in enumerate(pivot.index):
        for col_idx, dataset_name in enumerate(pivot.columns):
            value = pivot.loc[feature, dataset_name]
            if math.isfinite(float(value)):
                ax.text(col_idx, row_idx, sign_label(float(value)), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Cohen's d")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def auc_ci_bootstrap(labels: np.ndarray, probs: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(int(n_boot)):
        indices = rng.integers(0, n, size=n)
        sampled_labels = labels[indices]
        sampled_probs = probs[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        aucs.append(float(roc_auc_score(sampled_labels, sampled_probs)))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def auc_permutation_pvalue(
    labels: np.ndarray,
    probs: np.ndarray,
    n_perm: int,
    seed: int,
    alternative: str,
) -> float:
    rng = np.random.default_rng(seed)
    observed = float(roc_auc_score(labels, probs))
    perm_values = []
    for _ in range(int(n_perm)):
        perm_labels = labels[rng.permutation(len(labels))]
        if np.unique(perm_labels).size < 2:
            continue
        perm_values.append(float(roc_auc_score(perm_labels, probs)))
    if not perm_values:
        return float("nan")
    perm_arr = np.asarray(perm_values, dtype=np.float64)
    if alternative == "greater":
        extreme = np.sum(perm_arr >= observed)
    elif alternative == "less":
        extreme = np.sum(perm_arr <= observed)
    else:
        extreme = np.sum(np.abs(perm_arr - 0.5) >= abs(observed - 0.5))
    return float((extreme + 1) / (len(perm_arr) + 1))


def one_sided_ttest(values: np.ndarray, mu: float, alternative: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan")
    result = stats.ttest_1samp(arr, popmean=mu, nan_policy="omit")
    if not math.isfinite(float(result.pvalue)):
        return float("nan")
    statistic = float(result.statistic)
    p_two = float(result.pvalue)
    if alternative == "greater":
        return p_two / 2.0 if statistic > 0 else 1.0 - p_two / 2.0
    if alternative == "less":
        return p_two / 2.0 if statistic < 0 else 1.0 - p_two / 2.0
    return p_two


def analyze_prediction_files(
    dataset_name: str,
    prediction_dir: Path,
    prob_column: str,
    flip_labels: bool,
    bootstrap_iters: int,
    permutation_iters: int,
    seed_base: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for csv_path in sorted(prediction_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        strategy = str(df["strategy"].iloc[0])
        seed = int(df["seed"].iloc[0])
        labels = df["label"].to_numpy(dtype=np.int64)
        probs = df[prob_column].to_numpy(dtype=np.float64)
        if flip_labels:
            labels = 1 - labels
        if np.unique(labels).size < 2:
            continue
        auc = float(roc_auc_score(labels, probs))
        ci_low, ci_high = auc_ci_bootstrap(labels, probs, bootstrap_iters, seed_base + seed)
        if flip_labels:
            alternative = "greater"
        elif dataset_name == "ds007216":
            alternative = "less"
        else:
            alternative = "greater"
        perm_p = auc_permutation_pvalue(labels, probs, permutation_iters, seed_base + seed + 17, alternative)
        rows.append(
            {
                "dataset": dataset_name,
                "label_view": "flip" if flip_labels else "original",
                "strategy": strategy,
                "seed": seed,
                "n_subjects": int(len(df)),
                "roc_auc": auc,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_p": perm_p,
                "auc_minus_chance": auc - 0.5,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "label_view", "strategy", "seed"]).reset_index(drop=True)


def summarize_prediction_tests(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset_name, label_view, strategy), sub_df in seed_df.groupby(["dataset", "label_view", "strategy"]):
        aucs = sub_df["roc_auc"].to_numpy(dtype=np.float64)
        diffs = aucs - 0.5
        if label_view == "flip" or dataset_name == "mendeley":
            alternative = "greater"
        elif dataset_name == "ds007216":
            alternative = "less"
        else:
            alternative = "greater"
        try:
            wilcoxon_p = float(stats.wilcoxon(diffs, alternative=alternative).pvalue)
        except Exception:
            wilcoxon_p = float("nan")
        fisher_p = float(stats.combine_pvalues(sub_df["permutation_p"].to_numpy(dtype=np.float64), method="fisher")[1])
        rows.append(
            {
                "dataset": dataset_name,
                "label_view": label_view,
                "strategy": strategy,
                "n_seeds": int(len(sub_df)),
                "roc_auc_mean": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "one_sided_ttest_p": one_sided_ttest(aucs, 0.5, alternative),
                "wilcoxon_p": wilcoxon_p,
                "fisher_permutation_p": fisher_p,
                "seed_auc_min": float(np.min(aucs)),
                "seed_auc_median": float(np.median(aucs)),
                "seed_auc_max": float(np.max(aucs)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "label_view", "strategy"]).reset_index(drop=True)


def build_report(
    direction_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    prediction_summary_df: pd.DataFrame,
    reverse_seed_df: pd.DataFrame,
) -> str:
    sign_pivot = (
        direction_df.pivot(index="feature", columns="dataset", values="direction_by_diff")
        .reindex(index=CORE_FEATURES, columns=HEATMAP_DATASETS)
        .fillna("0")
    )
    report_lines = [
        "# Direction Consistency Analysis",
        "",
        "## Shared common feature directions",
        sign_pivot.to_markdown(),
        "",
        "## ds007216 reversal summary",
        consistency_df[
            [
                "feature",
                "reference_direction",
                "mendeley_direction",
                "ds007216_direction",
                "ds007216_matches_reference",
                "ds007216_reverses_reference",
            ]
        ].to_markdown(index=False),
        "",
        "## External AUC significance",
        prediction_summary_df.to_markdown(index=False),
        "",
        "## Existing reverse-discrimination seed evidence",
        reverse_seed_df[
            [
                "seed",
                "strategy",
                "orig_roc_auc",
                "flip_roc_auc",
                "auc_gain_after_flip",
                "seed_supports_flip",
            ]
        ].to_markdown(index=False),
        "",
    ]
    return "\n".join(report_lines) + "\n"


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)
    dataset_csvs = dataset_csvs_from_root(args.features_root)
    dataset_csvs["mendeley"] = args.mendeley_features
    dataset_csvs["ds007216"] = args.ds007216_features

    direction_rows: list[dict] = []
    for dataset_name, csv_path in dataset_csvs.items():
        direction_rows.extend(build_direction_rows(dataset_name, csv_path, CORE_FEATURES))
    direction_df = pd.DataFrame(direction_rows)
    consistency_df = summarize_consistency(direction_df)

    direction_df.to_csv(output_root / "feature_directions.csv", index=False, encoding="utf-8-sig")
    consistency_df.to_csv(output_root / "feature_consistency_summary.csv", index=False, encoding="utf-8-sig")
    make_heatmap(direction_df, output_root / "feature_direction_heatmap.png")

    prediction_summary_df = pd.DataFrame()
    reverse_seed_df = pd.DataFrame()
    if not args.skip_prediction_tests:
        seed_tests = pd.concat(
            [
                analyze_prediction_files(
                    dataset_name="mendeley",
                    prediction_dir=args.mendeley_pred_dir,
                    prob_column="prob_high",
                    flip_labels=False,
                    bootstrap_iters=args.bootstrap_iters,
                    permutation_iters=args.permutation_iters,
                    seed_base=args.seed,
                ),
                analyze_prediction_files(
                    dataset_name="ds007216",
                    prediction_dir=args.ds007216_pred_dir,
                    prob_column="prob",
                    flip_labels=False,
                    bootstrap_iters=args.bootstrap_iters,
                    permutation_iters=args.permutation_iters,
                    seed_base=args.seed + 1000,
                ),
                analyze_prediction_files(
                    dataset_name="ds007216",
                    prediction_dir=args.ds007216_pred_dir,
                    prob_column="prob",
                    flip_labels=True,
                    bootstrap_iters=args.bootstrap_iters,
                    permutation_iters=args.permutation_iters,
                    seed_base=args.seed + 2000,
                ),
            ],
            ignore_index=True,
        )
        prediction_summary_df = summarize_prediction_tests(seed_tests)
        seed_tests.to_csv(output_root / "external_auc_seed_tests.csv", index=False, encoding="utf-8-sig")
        prediction_summary_df.to_csv(output_root / "external_auc_group_tests.csv", index=False, encoding="utf-8-sig")

        reverse_seed_df = pd.read_csv(args.ds007216_reverse_csv)
        reverse_seed_df.to_csv(output_root / "ds007216_reverse_seed_evidence.csv", index=False, encoding="utf-8-sig")

        report_text = build_report(direction_df, consistency_df, prediction_summary_df, reverse_seed_df)
        (output_root / "direction_consistency_report.md").write_text(report_text, encoding="utf-8")

    metadata = {
        "core_features": CORE_FEATURES,
        "datasets": list(dataset_csvs.keys()),
        "features_root": str(Path(args.features_root)),
        "skip_prediction_tests": bool(args.skip_prediction_tests),
        "bootstrap_iters": int(args.bootstrap_iters),
        "permutation_iters": int(args.permutation_iters),
    }
    (output_root / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved direction-consistency analysis to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
