"""中文说明

用途：
    比较本地前期数据在 segment-random 与 subject-independent 划分下的指标差异，
    用于说明片段随机划分可能高估 EEG 焦虑识别性能。
输入：
    本地授权原始 FIF/标签数据；该数据不随公开仓库分发。
输出：
    两种划分协议下的多 seed 指标、预测表和协议对照 summary。
快速运行：
    需要先在本地提供 legacy helper `anxiety_eeg.legacy.local_anxiety_dataset`；
    然后运行 `python -m anxiety_eeg.analysis.compare_split_protocols --output-root outputs/split_protocol_comparison`。
论文对应：
    第 3 章无泄露评估协议和第 5 章 split-protocol control。
注意事项：
    这是本地原始数据审计脚本，不是公开仓库默认 smoke 流程。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "split_protocol_comparison"


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
DEFAULT_SEEDS = [42, 123, 224, 3407, 65422, 7, 21, 2024, 31415, 27182]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare segment-random and subject-independent split protocols on local paired EEG windows."
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--windows-per-subject", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.30)
    return parser.parse_args()


def load_local_anxiety_helpers():
    try:
        from anxiety_eeg.legacy.local_anxiety_dataset import (  # type: ignore
            LOCAL_CHANNELS,
            TARGET_SR,
            WINDOW_SECONDS,
            collect_pair_records,
        )
    except ImportError as exc:
        raise RuntimeError(
            "缺少本地授权数据 helper：anxiety_eeg.legacy.local_anxiety_dataset。"
            "公开仓库默认不分发本地原始队列；如需运行 split-protocol 审计，"
            "请按 docs/legacy_preliminary.md 放置本地 helper 和原始数据路径。"
        ) from exc
    return LOCAL_CHANNELS, TARGET_SR, WINDOW_SECONDS, collect_pair_records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class WindowRow:
    subject: str
    source_split: str
    anxiety: float
    label: int
    window_index: int
    features: np.ndarray


class LocalWindowFeatureBuilder:
    def __init__(self, channels: tuple[str, ...], sample_rate: int, window_seconds: float):
        self.channels = list(channels)
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.raw_cache: dict[Path, object] = {}
        self.channel_cache: dict[Path, list[str]] = {}

    def _load_raw(self, path: Path):
        if path not in self.raw_cache:
            import mne

            raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(eeg_picks) == 0:
                raise ValueError(f"No EEG channels in {path}")
            raw.pick(eeg_picks)
            self.raw_cache[path] = raw
            self.channel_cache[path] = list(raw.ch_names)
        return self.raw_cache[path]

    def _resample_array(self, x_ct: np.ndarray, orig_sfreq: float) -> np.ndarray:
        if abs(orig_sfreq - self.sample_rate) < 1e-6:
            return x_ct.astype(np.float32)
        target_len = max(1, int(round(x_ct.shape[1] * self.sample_rate / orig_sfreq)))
        old_axis = np.linspace(0.0, 1.0, x_ct.shape[1], dtype=np.float32)
        new_axis = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        out = np.zeros((x_ct.shape[0], target_len), dtype=np.float32)
        for idx in range(x_ct.shape[0]):
            out[idx] = np.interp(new_axis, old_axis, x_ct[idx]).astype(np.float32)
        return out

    def _align_channels(self, x_ct: np.ndarray, ch_names: list[str]) -> np.ndarray:
        aligned = np.zeros((len(self.channels), x_ct.shape[1]), dtype=np.float32)
        name_to_idx = {name.lower(): idx for idx, name in enumerate(ch_names)}
        for target_idx, name in enumerate(self.channels):
            source_idx = name_to_idx.get(name.lower())
            if source_idx is not None:
                aligned[target_idx] = x_ct[source_idx]
        return aligned

    def _zscore(self, x_ct: np.ndarray) -> np.ndarray:
        mean = x_ct.mean(axis=1, keepdims=True)
        std = x_ct.std(axis=1, keepdims=True)
        return (x_ct - mean) / (std + 1e-6)

    def _safe_read_segment(self, raw, start: int, stop: int) -> np.ndarray | None:
        try:
            return raw.get_data(start=int(start), stop=int(stop)).astype(np.float32)
        except Exception:
            return None

    def _extract_fixed_windows(self, path: Path, n_windows: int) -> list[np.ndarray]:
        raw = self._load_raw(path)
        sfreq = float(raw.info["sfreq"])
        total_len = int(raw.n_times)
        win_len = max(1, int(round(self.window_seconds * sfreq)))
        max_start = max(0, total_len - win_len)
        if max_start == 0:
            starts = [0]
        else:
            starts = np.linspace(0, max_start, num=min(int(n_windows), max_start + 1), dtype=int).tolist()
        segments: list[np.ndarray] = []
        for start in starts:
            stop = min(total_len, start + win_len)
            fallback_starts = [int(start), 0, max_start // 2, max_start]
            x_ct = None
            seen: set[int] = set()
            for fallback_start in fallback_starts:
                fallback_start = int(max(0, min(max_start, fallback_start)))
                if fallback_start in seen:
                    continue
                seen.add(fallback_start)
                fallback_stop = min(total_len, fallback_start + win_len)
                x_ct = self._safe_read_segment(raw, fallback_start, fallback_stop)
                if x_ct is not None and x_ct.size > 0:
                    break
            if x_ct is None or x_ct.size == 0:
                continue
            x_ct = self._resample_array(x_ct, sfreq)
            x_ct = self._align_channels(x_ct, self.channel_cache[path])
            segments.append(self._zscore(x_ct))
        if not segments:
            raise RuntimeError(f"Could not extract any valid windows from {path}")
        return segments

    def _relative_band_powers(self, x_ct: np.ndarray) -> np.ndarray:
        freqs, psd = welch(
            x_ct,
            fs=float(self.sample_rate),
            nperseg=min(x_ct.shape[1], int(self.sample_rate * 2)),
            axis=1,
        )
        total_mask = (freqs >= 1.0) & (freqs <= 45.0)
        total_power = np.sum(psd[:, total_mask], axis=1, keepdims=True) + 1e-8
        band_vectors = []
        for low, high in BANDS.values():
            band_mask = (freqs >= low) & (freqs < high)
            band_power = np.sum(psd[:, band_mask], axis=1, keepdims=True)
            band_vectors.append((band_power / total_power).astype(np.float32))
        return np.concatenate(band_vectors, axis=1)

    def paired_window_features(self, target_path: Path, baseline_path: Path, n_windows: int) -> list[np.ndarray]:
        target_windows = self._extract_fixed_windows(target_path, n_windows)
        baseline_windows = self._extract_fixed_windows(baseline_path, n_windows)
        total = min(len(target_windows), len(baseline_windows))
        output: list[np.ndarray] = []
        for idx in range(total):
            target_bands = self._relative_band_powers(target_windows[idx])
            baseline_bands = self._relative_band_powers(baseline_windows[idx])
            diff_bands = target_bands - baseline_bands
            target_global = np.mean(target_bands, axis=0)
            baseline_global = np.mean(baseline_bands, axis=0)
            diff_global = target_global - baseline_global
            feature_vector = np.concatenate(
                [
                    target_bands.reshape(-1),
                    baseline_bands.reshape(-1),
                    diff_bands.reshape(-1),
                    target_global,
                    baseline_global,
                    diff_global,
                ]
            ).astype(np.float32)
            output.append(feature_vector)
        return output


def build_subject_split(pair_records: list, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    ordered = sorted(pair_records, key=lambda row: (float(row.tra_anx), row.subject))
    n_bins = min(10, max(4, len(ordered) // 8 or 1))
    rng = np.random.default_rng(int(seed))
    train_subjects: list[str] = []
    val_subjects: list[str] = []
    for bucket in np.array_split(np.asarray(ordered, dtype=object), n_bins):
        bucket_rows = list(bucket.tolist())
        rng.shuffle(bucket_rows)
        if len(bucket_rows) == 1:
            val_count = 0
        else:
            val_count = int(round(len(bucket_rows) * float(val_fraction)))
            val_count = max(1, min(len(bucket_rows) - 1, val_count))
        val_subjects.extend([row.subject for row in bucket_rows[:val_count]])
        train_subjects.extend([row.subject for row in bucket_rows[val_count:]])
    return set(train_subjects), set(val_subjects)


def safe_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) < 3 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def metrics_dict(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": safe_roc_auc(labels, probs),
    }


def fit_and_score(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, float]:
    feature_cols = [col for col in train_df.columns if col.startswith("f_")]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float64))
    x_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float64))
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(x_train, y_train)
    val_probs = model.predict_proba(x_val)[:, 1]

    window_metrics = metrics_dict(y_val, val_probs)
    val_subject_df = (
        val_df.assign(prob=val_probs)
        .groupby(["subject", "label"], as_index=False)["prob"]
        .mean()
        .rename(columns={"prob": "subject_prob"})
    )
    subject_metrics = metrics_dict(
        val_subject_df["label"].to_numpy(dtype=np.int64),
        val_subject_df["subject_prob"].to_numpy(dtype=np.float64),
    )

    return {
        "window_accuracy": window_metrics["accuracy"],
        "window_balanced_accuracy": window_metrics["balanced_accuracy"],
        "window_f1": window_metrics["f1"],
        "window_roc_auc": window_metrics["roc_auc"],
        "subject_accuracy": subject_metrics["accuracy"],
        "subject_balanced_accuracy": subject_metrics["balanced_accuracy"],
        "subject_f1": subject_metrics["f1"],
        "subject_roc_auc": subject_metrics["roc_auc"],
        "n_train_windows": int(len(train_df)),
        "n_val_windows": int(len(val_df)),
        "n_train_subjects": int(train_df["subject"].nunique()),
        "n_val_subjects": int(val_df["subject"].nunique()),
    }


def aggregate_protocol_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metric_cols = [
        "window_roc_auc",
        "window_balanced_accuracy",
        "subject_roc_auc",
        "subject_balanced_accuracy",
        "subject_overlap_ratio",
    ]
    for protocol, sub_df in seed_df.groupby("protocol"):
        row = {"protocol": protocol, "n_seeds": int(len(sub_df))}
        for metric in metric_cols:
            values = sub_df[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("protocol").reset_index(drop=True)


def plot_protocol_results(agg_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["subject_roc_auc", "subject_balanced_accuracy"]
    titles = ["Subject-level ROC-AUC", "Subject-level Balanced Accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    protocols = agg_df["protocol"].tolist()
    x = np.arange(len(protocols))
    for axis, metric, title in zip(axes, metrics, titles):
        means = agg_df[f"{metric}_mean"].to_numpy(dtype=np.float64)
        stds = agg_df[f"{metric}_std"].to_numpy(dtype=np.float64)
        axis.bar(x, means, yerr=stds, capsize=4, color=["#d95f02", "#1b9e77"])
        axis.set_xticks(x)
        axis.set_xticklabels(protocols, rotation=20, ha="right")
        axis.set_ylim(0.0, max(0.75, float(np.max(means + stds)) + 0.05))
        axis.set_title(title)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_window_feature_table(window_rows: list[WindowRow]) -> pd.DataFrame:
    feature_count = int(window_rows[0].features.shape[0])
    records = []
    for row in window_rows:
        payload = {
            "subject": row.subject,
            "source_split": row.source_split,
            "anxiety": float(row.anxiety),
            "label": int(row.label),
            "window_index": int(row.window_index),
        }
        for idx in range(feature_count):
            payload[f"f_{idx:03d}"] = float(row.features[idx])
        records.append(payload)
    return pd.DataFrame(records)


def build_report(agg_df: pd.DataFrame, seed_df: pd.DataFrame, global_threshold: float) -> str:
    header = [
        "# Split Protocol Comparison",
        "",
        f"- Fixed cohort median threshold used for both protocols: `{global_threshold:.3f}`",
        "- Model: logistic regression on paired stress-vs-neutral spectral window features.",
        "- Purpose: isolate how split protocol alone changes apparent performance.",
        "",
        "## Aggregate results",
        agg_df.to_markdown(index=False),
        "",
        "## Seed-level results",
        seed_df.to_markdown(index=False),
        "",
    ]
    return "\n".join(header) + "\n"


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)
    local_channels, target_sr, window_seconds, collect_pair_records = load_local_anxiety_helpers()

    pair_records = collect_pair_records()
    global_threshold = float(np.median([float(row.tra_anx) for row in pair_records]))
    feature_builder = LocalWindowFeatureBuilder(local_channels, target_sr, window_seconds)

    window_rows: list[WindowRow] = []
    skipped_subjects: list[str] = []
    for pair in pair_records:
        label = int(float(pair.tra_anx) >= global_threshold)
        try:
            paired_features = feature_builder.paired_window_features(
                target_path=pair.target_path,
                baseline_path=pair.baseline_path,
                n_windows=args.windows_per_subject,
            )
        except Exception:
            skipped_subjects.append(pair.subject)
            continue
        for window_index, feature_vector in enumerate(paired_features):
            window_rows.append(
                WindowRow(
                    subject=pair.subject,
                    source_split=pair.split,
                    anxiety=float(pair.tra_anx),
                    label=label,
                    window_index=window_index,
                    features=feature_vector,
                )
            )

    window_df = build_window_feature_table(window_rows)
    seed_rows: list[dict] = []

    for seed in args.seeds:
        train_subjects, val_subjects = build_subject_split(pair_records, args.val_fraction, seed)
        subject_train_df = window_df[window_df["subject"].isin(train_subjects)].reset_index(drop=True)
        subject_val_df = window_df[window_df["subject"].isin(val_subjects)].reset_index(drop=True)
        subject_metrics = fit_and_score(subject_train_df, subject_val_df)
        seed_rows.append(
            {
                "seed": int(seed),
                "protocol": "subject_independent",
                "subject_overlap_count": 0,
                "subject_overlap_ratio": 0.0,
                **subject_metrics,
            }
        )

        random_train_df, random_val_df = train_test_split(
            window_df,
            test_size=args.val_fraction,
            random_state=int(seed),
            stratify=window_df["label"],
        )
        random_train_df = random_train_df.reset_index(drop=True)
        random_val_df = random_val_df.reset_index(drop=True)
        overlap_subjects = set(random_train_df["subject"]).intersection(set(random_val_df["subject"]))
        random_metrics = fit_and_score(random_train_df, random_val_df)
        seed_rows.append(
            {
                "seed": int(seed),
                "protocol": "segment_random",
                "subject_overlap_count": int(len(overlap_subjects)),
                "subject_overlap_ratio": float(len(overlap_subjects) / max(random_val_df["subject"].nunique(), 1)),
                **random_metrics,
            }
        )

    seed_df = pd.DataFrame(seed_rows).sort_values(["protocol", "seed"]).reset_index(drop=True)
    agg_df = aggregate_protocol_results(seed_df)

    seed_df.to_csv(output_root / "split_protocol_seed_results.csv", index=False, encoding="utf-8-sig")
    agg_df.to_csv(output_root / "split_protocol_aggregate.csv", index=False, encoding="utf-8-sig")
    plot_protocol_results(agg_df, output_root / "split_protocol_comparison.png")
    report_text = build_report(agg_df, seed_df, global_threshold)
    (output_root / "split_protocol_report.md").write_text(report_text, encoding="utf-8")

    run_config = {
        "seeds": [int(seed) for seed in args.seeds],
        "windows_per_subject": int(args.windows_per_subject),
        "val_fraction": float(args.val_fraction),
        "global_threshold": global_threshold,
        "channels": list(local_channels),
        "sample_rate": int(target_sr),
        "window_seconds": float(window_seconds),
        "included_subjects": int(window_df["subject"].nunique()),
        "skipped_subjects": skipped_subjects,
    }
    (output_root / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved split-protocol comparison to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
