#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文说明

用途：
    公开数据集和本地数据集的 EEG 频谱特征提取公共工具，输出主模型需要的
    `subject_features.csv`。
输入：
    原始 EEG 文件、量表/标签文件和数据集配置。
输出：
    subject-level global/regional/contrast spectral features 以及数据集审计文件。
快速运行：
    通常通过 `python -m anxiety_eeg.features.score_ds003478` 等入口调用。
论文对应：
    第 3 章数据与标签构建、第 4 章全局频谱组织表征。
注意事项：
    原始数据不随仓库分发；公开数据请按 `data/<dataset>/README.md` 下载。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import mne
    import numpy as np
    from scipy.signal import welch
    from scipy.stats import pearsonr, spearmanr
except ImportError as exc:  # pragma: no cover - depends on local env
    raise SystemExit(
        "Missing dependency. Activate the EEG environment before running these tools. "
        f"Import error: {exc}"
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DS_ROOT = REPO_ROOT / "data" / "ds003478"
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "original_local"
DEFAULT_LOCAL_CSV = DEFAULT_LOCAL_ROOT / "PI_emo_tra_anx.csv"


def _load_legacy_eeg_helpers():
    try:
        from anxiety_eeg.legacy.analyze_anxiety_correlations import (  # type: ignore
            compute_windowed_channel_features,
            load_ds_spans,
            load_ds_subjects,
            load_local_subjects,
            local_fif_paths,
            ranges_from_spans,
            select_eeg_channels,
        )
    except ImportError as exc:
        raise RuntimeError(
            "缺少本地原始 EEG 特征提取 helper。公开仓库默认只保留复现实验主线和"
            "受试者级特征接口；若要从原始 EEG 重建特征，请按 docs/reproduction.md "
            "补充 legacy helper 或使用已生成的 subject_features.csv。"
        ) from exc
    return {
        "compute_windowed_channel_features": compute_windowed_channel_features,
        "load_ds_spans": load_ds_spans,
        "load_ds_subjects": load_ds_subjects,
        "load_local_subjects": load_local_subjects,
        "local_fif_paths": local_fif_paths,
        "ranges_from_spans": ranges_from_spans,
        "select_eeg_channels": select_eeg_channels,
    }


EPS = 1e-8
RELATIVE_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
REGIONAL_BANDS = ("theta", "alpha", "beta")
PRIMARY_REGION_NAMES = ("frontal", "fronto_central", "central", "parietal")
BAND_CENTERS = {
    "delta": 2.5,
    "theta": 6.0,
    "alpha": 10.5,
    "beta": 21.5,
    "gamma": 37.5,
}
BAND_RANGES = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("beta_high", 20.0, 30.0),
    ("gamma", 30.0, 45.0),
)
BROADBAND = (1.0, 45.0)
DEFAULT_WINDOW_SEC = 20.0
DEFAULT_MAX_WINDOWS = 8
DEFAULT_NPERSEG_SEC = 2.0
DEFAULT_MIN_N = 10


def display_channel_name(name: str) -> str:
    upper = name.strip().upper()
    if upper.startswith("FP"):
        return "Fp" + upper[2:]
    if upper.endswith("Z"):
        return upper[:-1] + "z"
    return upper


def select_eeg_channels(raw, common_local_only: bool) -> tuple[list[int], list[str]]:
    non_eeg_channels = {"HEOG", "VEOG", "EKG", "ECG", "STATUS", "TRIGGER"}
    common_local_channels = {"fp1", "fp2", "af3", "af4", "fpz", "f3", "f4"}
    picks = []
    names = []
    for idx, raw_name in enumerate(raw.ch_names):
        pretty = display_channel_name(raw_name)
        if pretty.upper() in non_eeg_channels:
            continue
        if common_local_only and pretty.lower() not in common_local_channels:
            continue
        picks.append(idx)
        names.append(pretty)
    if not picks:
        raise ValueError("No EEG channels selected")
    return picks, names


def integrate_band(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return np.trapezoid(psd[:, mask], freqs[mask], axis=1)
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def deterministic_windows(
    ranges: list[tuple[int, int]],
    win_len: int,
    max_windows: int,
) -> list[tuple[int, int]]:
    candidates = []
    for start, stop in ranges:
        latest = stop - win_len
        if latest < start:
            continue
        starts = [start + (latest - start) // 2] if max_windows <= 1 else np.linspace(start, latest, max_windows)
        for value in starts:
            s = int(round(float(value)))
            candidates.append((s, s + win_len))
    if len(candidates) <= max_windows:
        return candidates
    keep = np.linspace(0, len(candidates) - 1, max_windows)
    return [candidates[int(round(float(i)))] for i in keep]


def compute_windowed_channel_features(
    raw,
    picks: list[int],
    channel_names: list[str],
    ranges: list[tuple[int, int]],
    window_sec: float,
    max_windows: int,
    nperseg_sec: float,
) -> dict[str, dict[str, float]]:
    sfreq = float(raw.info["sfreq"])
    win_len = max(1, int(round(float(window_sec) * sfreq)))
    windows = deterministic_windows(ranges, win_len=win_len, max_windows=max_windows)
    if not windows:
        windows = [(0, min(raw.n_times, win_len))]
    accum: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in channel_names
    }
    for start, stop in windows:
        data = raw.get_data(picks=picks, start=int(start), stop=int(stop))
        if data.size == 0:
            continue
        freqs, psd = welch(
            data,
            fs=sfreq,
            nperseg=min(data.shape[1], max(8, int(round(float(nperseg_sec) * sfreq)))),
            axis=1,
        )
        broad = integrate_band(freqs, psd, BROADBAND[0], BROADBAND[1]) + EPS
        for band_name, low, high in BAND_RANGES:
            power = integrate_band(freqs, psd, low, high)
            rel = power / broad
            for idx, ch_name in enumerate(channel_names):
                accum[ch_name][f"log_{band_name}"].append(float(np.log(power[idx] + EPS)))
                accum[ch_name][f"rel_{band_name}"].append(float(rel[idx]))
        for idx, ch_name in enumerate(channel_names):
            accum[ch_name]["log_broadband"].append(float(np.log(broad[idx] + EPS)))
    return {
        ch_name: {feature: float(np.mean(values)) for feature, values in feature_map.items()}
        for ch_name, feature_map in accum.items()
    }


@dataclass
class SubjectFeatureBundle:
    dataset_name: str
    score_name: str
    subject: str
    split: str
    anxiety: float
    context: str
    channel_names: list[str]
    channel_features: dict[str, dict[str, float]]
    source_count: int
    metadata: dict[str, object]


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--nperseg-sec", type=float, default=DEFAULT_NPERSEG_SEC)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument(
        "--use-common-local-channels",
        action="store_true",
        help="Restrict channel selection to Fp1/Fp2/AF3/AF4/Fpz/F3/F4.",
    )
    return parser


def canonical_channel_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return text
    upper = text.upper()
    if upper.startswith("FP"):
        return "Fp" + upper[2:]
    if upper.endswith("Z"):
        return upper[:-1] + "z"
    return upper


def canonicalize_channel_positions(channel_positions: dict[str, object] | None) -> dict[str, tuple[float, float, float]]:
    if not isinstance(channel_positions, dict):
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for raw_name, raw_pos in channel_positions.items():
        if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) < 3:
            continue
        try:
            xyz = tuple(float(raw_pos[idx]) for idx in range(3))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in xyz):
            continue
        if max(abs(value) for value in xyz) <= EPS:
            continue
        out[canonical_channel_name(raw_name)] = xyz
    return out


def extract_channel_positions(raw, picks: list[int], channel_names: list[str]) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for pick, raw_name in zip(picks, channel_names):
        loc = raw.info["chs"][pick]["loc"][:3]
        try:
            xyz = [float(loc[idx]) for idx in range(3)]
        except (TypeError, ValueError, IndexError):
            continue
        if not all(math.isfinite(value) for value in xyz):
            continue
        if max(abs(value) for value in xyz) <= EPS:
            continue
        positions[canonical_channel_name(raw_name)] = xyz
    return positions


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if x.size < 3 or y.size < 3:
        return float("nan"), float("nan")
    if np.unique(np.round(x, 12)).size < 2 or np.unique(np.round(y, 12)).size < 2:
        return float("nan"), float("nan")
    if method == "pearson":
        coef, pvalue = pearsonr(x, y)
    else:
        coef, pvalue = spearmanr(x, y)
    return float(coef), float(pvalue)


def standardized_beta(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= EPS or y_std <= EPS:
        return float("nan")
    x_z = (x - float(np.mean(x))) / x_std
    y_z = (y - float(np.mean(y))) / y_std
    denom = float(np.dot(x_z, x_z))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(x_z, y_z) / denom)


def cohens_d(high_values: np.ndarray, low_values: np.ndarray) -> float:
    if high_values.size < 2 or low_values.size < 2:
        return float("nan")
    mean_high = float(np.mean(high_values))
    mean_low = float(np.mean(low_values))
    var_high = float(np.var(high_values, ddof=1))
    var_low = float(np.var(low_values, ddof=1))
    denom_df = (high_values.size - 1) + (low_values.size - 1)
    if denom_df <= 0:
        return float("nan")
    pooled = ((high_values.size - 1) * var_high + (low_values.size - 1) * var_low) / denom_df
    if pooled <= EPS:
        return float("nan")
    return (mean_high - mean_low) / math.sqrt(pooled)


def infer_side(name: str) -> str:
    label = canonical_channel_name(name)
    upper = label.upper()
    if upper.endswith("Z"):
        return "midline"
    digits = "".join(ch for ch in upper if ch.isdigit())
    if not digits:
        return "unknown"
    return "left" if int(digits[-1]) % 2 == 1 else "right"


def is_frontal(name: str) -> bool:
    upper = canonical_channel_name(name).upper()
    return upper.startswith("FP") or upper.startswith("AF") or (
        upper.startswith("F") and not upper.startswith(("FC", "FT"))
    )


def is_fronto_central(name: str) -> bool:
    upper = canonical_channel_name(name).upper()
    return upper.startswith("FC") or upper.startswith("FT") or upper in {"FZ", "CZ"}


def is_central(name: str) -> bool:
    upper = canonical_channel_name(name).upper()
    return upper.startswith("C") and not upper.startswith(("CP", "CB"))


def is_parietal(name: str) -> bool:
    upper = canonical_channel_name(name).upper()
    return upper.startswith("P") or upper.startswith("PO") or upper.startswith("O")


def classify_regions_from_names(channel_names: list[str]) -> dict[str, list[str]]:
    regions = defaultdict(list)
    for raw_name in channel_names:
        name = canonical_channel_name(raw_name)
        if is_frontal(name):
            regions["frontal"].append(name)
            side = infer_side(name)
            if side == "left":
                regions["left_frontal"].append(name)
                regions["lateral_frontal"].append(name)
            elif side == "right":
                regions["right_frontal"].append(name)
                regions["lateral_frontal"].append(name)
            elif side == "midline":
                regions["midline_frontal"].append(name)
        if is_fronto_central(name):
            regions["fronto_central"].append(name)
            regions["anterior"].append(name)
        if is_central(name):
            regions["central"].append(name)
        if is_parietal(name):
            regions["parietal"].append(name)
            regions["posterior"].append(name)
        if is_frontal(name):
            regions["anterior"].append(name)
    return {key: sorted(set(values)) for key, values in regions.items() if values}


def classify_regions_from_positions(
    channel_names: list[str],
    channel_positions: dict[str, tuple[float, float, float]],
) -> dict[str, list[str]]:
    entries = []
    for raw_name in channel_names:
        name = canonical_channel_name(raw_name)
        xyz = channel_positions.get(name)
        if xyz is None:
            continue
        x, y, z = xyz
        entries.append((name, x, y, z))
    if len(entries) < 8:
        return {}

    xs = np.asarray([entry[1] for entry in entries], dtype=np.float64)
    ys = np.asarray([entry[2] for entry in entries], dtype=np.float64)
    zs = np.asarray([entry[3] for entry in entries], dtype=np.float64)
    abs_x = np.abs(xs)

    y_q30, y_q40, y_q50, y_q60, y_q70 = np.quantile(ys, [0.30, 0.40, 0.50, 0.60, 0.70])
    z_q30, z_q60 = np.quantile(zs, [0.30, 0.60])

    top_scalp_mask = zs >= z_q30
    frontal_mask = top_scalp_mask & (ys >= y_q70)
    fronto_central_mask = top_scalp_mask & (ys >= y_q50) & (ys < y_q70)
    central_mask = (zs >= z_q60) & (ys >= y_q40) & (ys <= y_q60)
    parietal_mask = top_scalp_mask & (ys <= y_q30)
    anterior_mask = top_scalp_mask & (ys >= y_q60)
    posterior_mask = top_scalp_mask & (ys <= y_q40)

    frontal_abs_x = abs_x[frontal_mask]
    if frontal_abs_x.size >= 6:
        midline_threshold = float(np.quantile(frontal_abs_x, 0.25))
        lateral_threshold = float(np.quantile(frontal_abs_x, 0.60))
    else:
        midline_threshold = float(np.quantile(abs_x, 0.20))
        lateral_threshold = float(np.quantile(abs_x, 0.60))

    regions = defaultdict(list)
    for idx, (name, x, _y, _z) in enumerate(entries):
        if frontal_mask[idx]:
            regions["frontal"].append(name)
            if abs(x) <= midline_threshold:
                regions["midline_frontal"].append(name)
            elif x < 0:
                regions["left_frontal"].append(name)
            elif x > 0:
                regions["right_frontal"].append(name)
            if abs(x) >= lateral_threshold:
                regions["lateral_frontal"].append(name)
        if fronto_central_mask[idx]:
            regions["fronto_central"].append(name)
        if central_mask[idx]:
            regions["central"].append(name)
        if parietal_mask[idx]:
            regions["parietal"].append(name)
        if anterior_mask[idx]:
            regions["anterior"].append(name)
        if posterior_mask[idx]:
            regions["posterior"].append(name)
    return {key: sorted(set(values)) for key, values in regions.items() if values}


def classify_regions(
    channel_names: list[str],
    channel_positions: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, list[str]]:
    regions = classify_regions_from_names(channel_names)
    named_primary_count = sum(1 for key in PRIMARY_REGION_NAMES if regions.get(key))
    if channel_positions is None or named_primary_count >= 2:
        return regions

    position_regions = classify_regions_from_positions(channel_names, channel_positions)
    if not position_regions:
        return regions
    for key, values in position_regions.items():
        if not regions.get(key):
            regions[key] = values
    return {key: sorted(set(values)) for key, values in regions.items() if values}


def get_feature_value(channel_features: dict[str, dict[str, float]], channels: list[str], key: str) -> float | None:
    values = []
    for channel in channels:
        value = channel_features.get(channel, {}).get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def region_relative_power(
    channel_features: dict[str, dict[str, float]],
    channels: list[str],
    band: str,
) -> float | None:
    return get_feature_value(channel_features, channels, f"rel_{band}")


def region_band_power(
    channel_features: dict[str, dict[str, float]],
    channels: list[str],
    band: str,
) -> float | None:
    value = get_feature_value(channel_features, channels, f"log_{band}")
    if value is None:
        return None
    return float(math.exp(value))


def safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or not math.isfinite(num) or not math.isfinite(den):
        return None
    if abs(den) <= EPS:
        return None
    return float(num / den)


def spectral_centroid_from_rel(rel_power: dict[str, float | None]) -> float | None:
    num = 0.0
    den = 0.0
    for band, center in BAND_CENTERS.items():
        value = rel_power.get(band)
        if value is None or not math.isfinite(value):
            continue
        num += center * float(value)
        den += float(value)
    if den <= EPS:
        return None
    return num / den


def one_over_f_slope_from_power(abs_power: dict[str, float | None]) -> float | None:
    xs = []
    ys = []
    for band, center in BAND_CENTERS.items():
        value = abs_power.get(band)
        if value is None or not math.isfinite(value) or value <= EPS:
            continue
        xs.append(math.log(center))
        ys.append(math.log(value))
    if len(xs) < 2:
        return None
    slope, _ = np.polyfit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), 1)
    return float(slope)


def build_candidate_features(bundle: SubjectFeatureBundle) -> dict[str, float]:
    channel_features = bundle.channel_features
    all_channels = [canonical_channel_name(name) for name in bundle.channel_names]
    channel_positions = canonicalize_channel_positions(bundle.metadata.get("channel_positions"))
    regions = classify_regions(all_channels, channel_positions=channel_positions)

    features: dict[str, float] = {}
    global_rel = {band: region_relative_power(channel_features, all_channels, band) for band in RELATIVE_BANDS}
    global_abs = {band: region_band_power(channel_features, all_channels, band) for band in RELATIVE_BANDS}

    for band in RELATIVE_BANDS:
        value = global_rel.get(band)
        if value is not None:
            features[f"global_rel_{band}"] = value

    beta_alpha = safe_ratio(global_abs.get("beta"), global_abs.get("alpha"))
    if beta_alpha is not None:
        features["global_beta_alpha_ratio"] = beta_alpha

    theta_beta = safe_ratio(global_abs.get("theta"), global_abs.get("beta"))
    if theta_beta is not None:
        features["global_theta_beta_ratio"] = theta_beta

    beta_delta_theta = safe_ratio(
        global_abs.get("beta"),
        None
        if global_abs.get("delta") is None or global_abs.get("theta") is None
        else float(global_abs["delta"] + global_abs["theta"]),
    )
    if beta_delta_theta is not None:
        features["global_beta_over_delta_theta"] = beta_delta_theta

    centroid = spectral_centroid_from_rel(global_rel)
    if centroid is not None:
        features["global_spectral_centroid"] = centroid

    slope = one_over_f_slope_from_power(global_abs)
    if slope is not None:
        features["global_one_over_f_slope"] = slope

    for region_name in ("frontal", "fronto_central", "central", "parietal"):
        region_channels = regions.get(region_name, [])
        if not region_channels:
            continue
        for band in REGIONAL_BANDS:
            value = region_relative_power(channel_features, region_channels, band)
            if value is not None:
                features[f"{region_name}_rel_{band}"] = value

    contrast_pairs = (
        ("frontal", "parietal", "contrast_frontal_minus_parietal"),
        ("anterior", "posterior", "contrast_anterior_minus_posterior"),
        ("left_frontal", "right_frontal", "contrast_left_frontal_minus_right_frontal"),
        ("midline_frontal", "lateral_frontal", "contrast_frontal_midline_minus_lateral"),
    )
    for left_region, right_region, prefix in contrast_pairs:
        left_channels = regions.get(left_region, [])
        right_channels = regions.get(right_region, [])
        if not left_channels or not right_channels:
            continue
        for band in REGIONAL_BANDS:
            left_value = region_relative_power(channel_features, left_channels, band)
            right_value = region_relative_power(channel_features, right_channels, band)
            if left_value is None or right_value is None:
                continue
            features[f"{prefix}_rel_{band}"] = float(left_value - right_value)

    return features


def build_subject_feature_rows(bundles: list[SubjectFeatureBundle]) -> tuple[list[dict], list[str], Counter]:
    feature_names = set()
    rows = []
    availability = Counter()
    for bundle in bundles:
        candidate_features = build_candidate_features(bundle)
        feature_names.update(candidate_features)
        for feature_name in candidate_features:
            availability[feature_name] += 1
        row = {
            "dataset": bundle.dataset_name,
            "score_name": bundle.score_name,
            "subject": bundle.subject,
            "split": bundle.split,
            "context": bundle.context,
            "anxiety": float(bundle.anxiety),
            "n_channels": len(bundle.channel_names),
            "source_count": int(bundle.source_count),
        }
        row.update(candidate_features)
        rows.append(row)
    return rows, sorted(feature_names), availability


def feature_family(feature_name: str) -> str:
    if feature_name.startswith("global_") and feature_name.endswith("_ratio"):
        return "global_ratio"
    if feature_name.startswith("global_spectral_") or feature_name.startswith("global_one_over_f_"):
        return "global_shape"
    if feature_name.startswith("global_rel_"):
        return "global_relative_power"
    if feature_name.startswith("contrast_"):
        return "regional_contrast"
    if "_rel_" in feature_name:
        return "regional_relative_power"
    return "other"


def score_subject_features(subject_rows: list[dict], feature_names: list[str], min_n: int) -> list[dict]:
    scores = []
    total_subjects = len(subject_rows)
    for feature_name in feature_names:
        values = []
        anxiety_values = []
        for row in subject_rows:
            value = row.get(feature_name)
            anxiety = row.get("anxiety")
            if value is None or anxiety is None:
                continue
            value = float(value)
            anxiety = float(anxiety)
            if not math.isfinite(value) or not math.isfinite(anxiety):
                continue
            values.append(value)
            anxiety_values.append(anxiety)

        n = len(values)
        if n < min_n:
            continue

        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(anxiety_values, dtype=np.float64)
        pearson_r, pearson_p = safe_corr(x, y, "pearson")
        spearman_r, spearman_p = safe_corr(x, y, "spearman")
        beta = standardized_beta(x, y)
        threshold = float(np.median(y))
        high_mask = y >= threshold
        low_mask = y < threshold
        high_values = x[high_mask]
        low_values = x[low_mask]
        effect_d = cohens_d(high_values, low_values)
        high_mean = float(np.mean(high_values)) if high_values.size else float("nan")
        low_mean = float(np.mean(low_values)) if low_values.size else float("nan")
        direction = (
            "positive"
            if math.isfinite(spearman_r) and spearman_r > 0
            else "negative"
            if math.isfinite(spearman_r) and spearman_r < 0
            else "flat"
        )
        priority_score = 100.0 * (
            0.50 * min(abs(spearman_r), 1.0) if math.isfinite(spearman_r) else 0.0
        )
        priority_score += 100.0 * (
            0.30 * min(abs(effect_d) / 1.5, 1.0) if math.isfinite(effect_d) else 0.0
        )
        priority_score += 100.0 * (
            0.20 * min(abs(beta), 1.0) if math.isfinite(beta) else 0.0
        )

        scores.append(
            {
                "feature": feature_name,
                "feature_family": feature_family(feature_name),
                "n": n,
                "coverage": float(n / total_subjects) if total_subjects else 0.0,
                "median_split_threshold": threshold,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "std_beta": beta,
                "cohens_d_median_split": effect_d,
                "high_group_mean": high_mean,
                "low_group_mean": low_mean,
                "direction": direction,
                "priority_score": float(priority_score),
            }
        )

    scores.sort(
        key=lambda row: (
            -row["priority_score"],
            -abs(row["spearman_r"]) if math.isfinite(row["spearman_r"]) else -1.0,
            row["feature"],
        )
    )
    for idx, row in enumerate(scores, start=1):
        row["rank"] = idx
    return scores


def build_dataset_info(
    dataset_name: str,
    score_name: str,
    bundles: list[SubjectFeatureBundle],
    subject_rows: list[dict],
    feature_availability: Counter,
    failures: list[dict],
    source_root: Path,
    output_dir: Path,
    config: dict[str, object],
) -> dict:
    split_counter = Counter()
    context_counter = Counter()
    channel_counter = Counter()
    region_counter = Counter()
    for bundle in bundles:
        split_counter[bundle.split] += 1
        context_counter[bundle.context] += 1
        for channel in bundle.channel_names:
            channel_counter[canonical_channel_name(channel)] += 1
        channel_positions = canonicalize_channel_positions(bundle.metadata.get("channel_positions"))
        for region_name in classify_regions(bundle.channel_names, channel_positions=channel_positions):
            region_counter[region_name] += 1

    return {
        "dataset_name": dataset_name,
        "score_name": score_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "successful_subjects": len(bundles),
        "subject_rows": len(subject_rows),
        "failures": len(failures),
        "splits": dict(sorted(split_counter.items())),
        "contexts": dict(sorted(context_counter.items())),
        "channel_coverage": dict(sorted(channel_counter.items())),
        "region_coverage": dict(sorted(region_counter.items())),
        "feature_availability": dict(sorted(feature_availability.items())),
        "config": config,
    }


def build_scorecard(
    dataset_name: str,
    score_name: str,
    feature_scores: list[dict],
    dataset_info: dict,
) -> dict:
    abs_rho = [abs(row["spearman_r"]) for row in feature_scores if math.isfinite(row["spearman_r"])]
    strong_count = sum(1 for row in feature_scores if row["priority_score"] >= 35.0)
    return {
        "dataset_name": dataset_name,
        "score_name": score_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "n_features_scored": len(feature_scores),
            "n_strong_features": strong_count,
            "max_abs_spearman_r": float(max(abs_rho)) if abs_rho else None,
            "median_abs_spearman_r": float(np.median(abs_rho)) if abs_rho else None,
        },
        "rank_rule": (
            "priority_score = 100 * (0.50*|spearman| + 0.30*clipped|cohens_d| + 0.20*|std_beta|)"
        ),
        "top_ranked_features": feature_scores[:15],
        "dataset_info_ref": str(Path(dataset_info["output_dir"]) / "dataset_info.json"),
    }


def build_text_report(dataset_info: dict, scorecard: dict, feature_scores: list[dict]) -> str:
    lines = [
        f"Dataset feasibility score report: {dataset_info['dataset_name']}",
        f"score_name              : {dataset_info['score_name']}",
        f"created_at              : {dataset_info['created_at']}",
        f"source_root             : {dataset_info['source_root']}",
        f"successful_subjects     : {dataset_info['successful_subjects']}",
        f"failures                : {dataset_info['failures']}",
        f"n_features_scored       : {scorecard['summary']['n_features_scored']}",
        f"n_strong_features       : {scorecard['summary']['n_strong_features']}",
        "",
        "Top features:",
    ]
    for row in feature_scores[:15]:
        lines.append(
            "  "
            f"#{row['rank']:02d} {row['feature']:<48} "
            f"rho={row['spearman_r']:+.3f} "
            f"d={row['cohens_d_median_split']:+.3f} "
            f"beta={row['std_beta']:+.3f} "
            f"score={row['priority_score']:.1f}"
        )
    return "\n".join(lines) + "\n"


def save_dataset_outputs(
    dataset_name: str,
    score_name: str,
    bundles: list[SubjectFeatureBundle],
    failures: list[dict],
    source_root: Path,
    output_dir: Path,
    config: dict[str, object],
    min_n: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_rows, feature_names, feature_availability = build_subject_feature_rows(bundles)
    feature_scores = score_subject_features(subject_rows, feature_names, min_n=min_n)
    dataset_info = build_dataset_info(
        dataset_name=dataset_name,
        score_name=score_name,
        bundles=bundles,
        subject_rows=subject_rows,
        feature_availability=feature_availability,
        failures=failures,
        source_root=source_root,
        output_dir=output_dir,
        config=config,
    )
    scorecard = build_scorecard(dataset_name, score_name, feature_scores, dataset_info)
    report = build_text_report(dataset_info, scorecard, feature_scores)

    write_csv(output_dir / "subject_features.csv", subject_rows)
    write_csv(output_dir / "feature_scores.csv", feature_scores)
    write_csv(output_dir / "failures.csv", failures)
    write_json(output_dir / "dataset_info.json", dataset_info)
    write_json(output_dir / "scorecard.json", scorecard)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")

    return {
        "dataset_info": dataset_info,
        "scorecard": scorecard,
        "subject_rows": subject_rows,
        "feature_scores": feature_scores,
        "failures": failures,
    }


def collect_ds003478_bundles(
    ds_root: Path,
    ds_run: str,
    ds_condition: str,
    window_sec: float,
    max_windows: int,
    nperseg_sec: float,
    use_common_local_channels: bool,
    limit_subjects: int | None = None,
) -> tuple[list[SubjectFeatureBundle], list[dict]]:
    helpers = _load_legacy_eeg_helpers()
    load_ds_subjects = helpers["load_ds_subjects"]
    load_ds_spans = helpers["load_ds_spans"]
    ranges_from_spans = helpers["ranges_from_spans"]
    select_eeg_channels = helpers["select_eeg_channels"]
    compute_windowed_channel_features = helpers["compute_windowed_channel_features"]
    bundles = []
    failures = []
    records = load_ds_subjects(ds_root)
    if limit_subjects is not None:
        records = records[:limit_subjects]
    print(f"[ds003478] subjects={len(records)} run={ds_run} condition={ds_condition}")

    for idx, rec in enumerate(records, start=1):
        stem = f"{rec.subject}_task-Rest_run-{ds_run}"
        set_path = ds_root / rec.subject / "eeg" / f"{stem}_eeg.set"
        events_path = ds_root / rec.subject / "eeg" / f"{stem}_events.tsv"
        if not set_path.exists() or not events_path.exists():
            reason = "missing_run_files"
            failures.append({"subject": rec.subject, "reason": reason})
            print(f"  [skip] {rec.subject}: {reason}")
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
            picks, names = select_eeg_channels(raw, common_local_only=use_common_local_channels)
            spans = load_ds_spans(events_path, raw.n_times / float(raw.info["sfreq"]))
            ranges = ranges_from_spans(spans, ds_condition, float(raw.info["sfreq"]), raw.n_times)
            features = compute_windowed_channel_features(
                raw=raw,
                picks=picks,
                channel_names=names,
                ranges=ranges,
                window_sec=window_sec,
                max_windows=max_windows,
                nperseg_sec=nperseg_sec,
            )
            bundles.append(
                SubjectFeatureBundle(
                    dataset_name="ds003478",
                    score_name="STAI",
                    subject=rec.subject,
                    split=rec.split,
                    anxiety=float(rec.anxiety),
                    context=ds_condition,
                    channel_names=list(features.keys()),
                    channel_features=features,
                    source_count=len(ranges),
                    metadata={"run": ds_run, "range_count": len(ranges)},
                )
            )
            print(f"  [{idx:03d}/{len(records):03d}] {rec.subject} channels={len(features)}")
        except Exception as exc:
            failures.append({"subject": rec.subject, "reason": repr(exc)})
            print(f"  [fail] {rec.subject}: {exc}")

    return bundles, failures


def collect_original_local_bundles(
    local_root: Path,
    local_csv: Path,
    emotions: list[str],
    window_sec: float,
    max_windows: int,
    nperseg_sec: float,
    use_common_local_channels: bool,
    limit_subjects: int | None = None,
) -> tuple[list[SubjectFeatureBundle], list[dict]]:
    helpers = _load_legacy_eeg_helpers()
    load_local_subjects = helpers["load_local_subjects"]
    local_fif_paths = helpers["local_fif_paths"]
    select_eeg_channels = helpers["select_eeg_channels"]
    compute_windowed_channel_features = helpers["compute_windowed_channel_features"]
    bundles = []
    failures = []
    records = load_local_subjects(local_csv)
    if limit_subjects is not None:
        records = records[:limit_subjects]
    print(f"[original_local] subjects={len(records)} emotions={emotions}")

    for idx, rec in enumerate(records, start=1):
        paths = local_fif_paths(local_root, rec, emotions)
        if not paths:
            reason = "no_fif_files"
            failures.append({"split": rec.split, "subject": rec.subject, "reason": reason})
            print(f"  [skip] {rec.split}/{rec.subject}: {reason}")
            continue

        accumulator: dict[tuple[str, str], list[float]] = defaultdict(list)
        channel_names = set()
        good_files = 0
        bad_files = 0
        for path in paths:
            try:
                raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
                picks, names = select_eeg_channels(raw, common_local_only=use_common_local_channels)
                features = compute_windowed_channel_features(
                    raw=raw,
                    picks=picks,
                    channel_names=names,
                    ranges=[(0, raw.n_times)],
                    window_sec=window_sec,
                    max_windows=max_windows,
                    nperseg_sec=nperseg_sec,
                )
                channel_names.update(features.keys())
                for channel, feature_dict in features.items():
                    for feature_name, value in feature_dict.items():
                        accumulator[(channel, feature_name)].append(float(value))
                good_files += 1
            except Exception as exc:
                bad_files += 1
                failures.append(
                    {
                        "split": rec.split,
                        "subject": rec.subject,
                        "file": path.name,
                        "reason": repr(exc),
                    }
                )
                print(f"    [bad file] {path.name}: {exc}")

        if not accumulator:
            print(f"  [skip] {rec.split}/{rec.subject}: no readable FIF files")
            continue

        merged_features: dict[str, dict[str, float]] = defaultdict(dict)
        for (channel, feature_name), values in accumulator.items():
            merged_features[channel][feature_name] = float(np.mean(values))

        bundles.append(
            SubjectFeatureBundle(
                dataset_name="original_local",
                score_name="tra_anx",
                subject=rec.subject,
                split=rec.split,
                anxiety=float(rec.anxiety),
                context="all_emotions" if "all" in emotions else "+".join(emotions),
                channel_names=sorted(channel_names),
                channel_features=dict(merged_features),
                source_count=good_files,
                metadata={"good_files": good_files, "bad_files": bad_files},
            )
        )
        print(
            f"  [{idx:03d}/{len(records):03d}] {rec.split}/{rec.subject} "
            f"files_ok={good_files}/{len(paths)} bad={bad_files}"
        )

    return bundles, failures


def default_scoring_dir(dataset_name: str) -> Path:
    return REPO_ROOT / "features" / "subject_features" / dataset_name


def run_dataset_pipeline(
    dataset_name: str,
    score_name: str,
    source_root: Path,
    output_dir: Path,
    min_n: int,
    config: dict[str, object],
    collector: Callable[[], tuple[list[SubjectFeatureBundle], list[dict]]],
) -> int:
    t0 = time.time()
    bundles, failures = collector()
    if not bundles:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "failures.csv", failures)
        write_json(
            output_dir / "dataset_info.json",
            {
                "dataset_name": dataset_name,
                "score_name": score_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_root": str(source_root),
                "output_dir": str(output_dir),
                "successful_subjects": 0,
                "failures": len(failures),
                "config": config,
            },
        )
        print(f"[{dataset_name}] No valid subjects were collected.")
        return 1

    outputs = save_dataset_outputs(
        dataset_name=dataset_name,
        score_name=score_name,
        bundles=bundles,
        failures=failures,
        source_root=source_root,
        output_dir=output_dir,
        config=config,
        min_n=min_n,
    )
    elapsed = time.time() - t0
    print(f"[{dataset_name}] subjects={len(outputs['subject_rows'])} features={len(outputs['feature_scores'])}")
    print(f"[{dataset_name}] output -> {output_dir}")
    print(f"[{dataset_name}] elapsed_sec={elapsed:.1f}")
    return 0


__all__ = [
    "DEFAULT_DS_ROOT",
    "DEFAULT_LOCAL_CSV",
    "DEFAULT_LOCAL_ROOT",
    "add_common_cli_arguments",
    "collect_ds003478_bundles",
    "collect_original_local_bundles",
    "default_scoring_dir",
    "run_dataset_pipeline",
]
