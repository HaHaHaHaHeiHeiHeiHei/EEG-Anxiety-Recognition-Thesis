"""中文说明

用途：
    构建论文主实验使用的受试者级 EEG 频谱特征数据集，支持三数据集
    `original_local`、`ds003478`、`ds007609` 的无泄露 subject-level 划分。
输入：
    每个数据集一个 `subject_features.csv`，默认位置为
    `features/subject_features/<dataset>/subject_features.csv`；smoke 测试可使用
    `tests/fixtures/subject_features`。
输出：
    PyTorch `Dataset`、训练/验证样本、每个 seed 的划分阈值与标准化信息。
快速运行：
    通常由 `python scripts/train_joint.py --features-root <特征根目录>` 间接调用。
论文对应：
    第 3 章“数据与标签构建”和第 4 章“EEG 全局频谱组织”。
注意事项：
    标签阈值、gray-zone 边界和标准化参数只由训练受试者估计，避免验证泄露。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES_ROOT = REPO_ROOT / "features" / "subject_features"
INTERNAL_DATASETS = ("original_local", "ds003478", "ds007609")


def dataset_csvs_from_root(
    features_root: Path | str,
    dataset_names: tuple[str, ...] | list[str] = INTERNAL_DATASETS,
) -> dict[str, Path]:
    """Return the canonical subject-feature CSV paths for a feature root."""
    root = Path(features_root)
    return {
        str(dataset_name): root / str(dataset_name) / "subject_features.csv"
        for dataset_name in dataset_names
    }


DEFAULT_DATASET_CSVS = dataset_csvs_from_root(DEFAULT_FEATURES_ROOT)

DEFAULT_GLOBAL_INPUT_FEATURES = [
    "global_rel_delta",
    "global_rel_theta",
    "global_rel_alpha",
    "global_rel_beta",
    "global_rel_gamma",
    "global_beta_alpha_ratio",
    "global_theta_beta_ratio",
    "global_beta_over_delta_theta",
    "global_spectral_centroid",
    "global_one_over_f_slope",
]
DEFAULT_REGION_INPUT_FEATURES = [
    "frontal_rel_theta",
    "frontal_rel_alpha",
    "frontal_rel_beta",
]
DEFAULT_EXPLORATORY_FEATURES = [
    "contrast_left_frontal_minus_right_frontal_rel_theta",
    "contrast_left_frontal_minus_right_frontal_rel_alpha",
    "contrast_left_frontal_minus_right_frontal_rel_beta",
    "contrast_frontal_midline_minus_lateral_rel_theta",
    "contrast_frontal_midline_minus_lateral_rel_alpha",
    "contrast_frontal_midline_minus_lateral_rel_beta",
]
DEFAULT_GLOBAL_CONSTRAINT_FEATURES = [
    "global_rel_beta",
    "global_beta_alpha_ratio",
    "global_beta_over_delta_theta",
    "global_spectral_centroid",
]
DEFAULT_REGION_CONSTRAINT_FEATURES = [
    "frontal_rel_beta",
]
SHARED_COMMON_INPUT_FEATURES = [
    "global_rel_theta",
    "global_rel_alpha",
    "global_rel_beta",
    "global_beta_alpha_ratio",
    "global_theta_beta_ratio",
    "global_spectral_centroid",
    "global_one_over_f_slope",
    "frontal_rel_theta",
    "frontal_rel_alpha",
    "frontal_rel_beta",
]
SHARED_COMMON_GLOBAL_CONSTRAINT_FEATURES = [
    "global_rel_beta",
    "global_beta_alpha_ratio",
    "global_spectral_centroid",
]
SHARED_COMMON_REGION_CONSTRAINT_FEATURES = [
    "frontal_rel_beta",
]

METADATA_COLUMNS = {
    "dataset",
    "score_name",
    "subject",
    "split",
    "context",
    "anxiety",
    "n_channels",
    "source_count",
}
EPS = 1e-8


def finite_float(value: object) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        raise ValueError(f"Cannot parse float from {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"Non-finite float value: {value!r}")
    return out


@dataclass
class SubjectRow:
    dataset: str
    score_name: str
    subject: str
    split: str
    context: str
    anxiety: float
    n_channels: int
    source_count: int
    features: dict[str, float]

    @property
    def subject_uid(self) -> str:
        return f"{self.dataset}::{self.subject}"


@dataclass
class DatasetThresholds:
    dataset: str
    score_name: str
    center: float
    std: float
    gray_z: float
    low_threshold: float
    high_threshold: float


@dataclass
class DatasetNormalizer:
    dataset: str
    feature_means: dict[str, float]
    feature_stds: dict[str, float]


@dataclass
class JointSample:
    dataset: str
    dataset_index: int
    subject: str
    subject_uid: str
    source_split: str
    score_name: str
    context: str
    anxiety: float
    label: int
    sample_weight: float
    in_gray_zone: bool
    x: np.ndarray
    global_target: np.ndarray
    region_target: np.ndarray


class JointConstraintDataset(Dataset):
    def __init__(self, samples: list[JointSample]):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("JointConstraintDataset needs at least one sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[int(index)]
        return {
            "x": torch.from_numpy(sample.x),
            "global_target": torch.from_numpy(sample.global_target),
            "region_target": torch.from_numpy(sample.region_target),
            "label": torch.tensor(float(sample.label), dtype=torch.float32),
            "sample_weight": torch.tensor(float(sample.sample_weight), dtype=torch.float32),
            "dataset_index": torch.tensor(int(sample.dataset_index), dtype=torch.long),
            "dataset": sample.dataset,
            "subject": sample.subject,
            "subject_uid": sample.subject_uid,
            "source_split": sample.source_split,
            "score_name": sample.score_name,
            "context": sample.context,
            "anxiety": torch.tensor(float(sample.anxiety), dtype=torch.float32),
            "in_gray_zone": torch.tensor(bool(sample.in_gray_zone), dtype=torch.bool),
        }


def read_subject_feature_csv(path: Path, dataset_name: str | None = None) -> list[SubjectRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in ("dataset", "subject", "anxiety") if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        feature_names = [name for name in fieldnames if name not in METADATA_COLUMNS]
        rows = []
        seen_subjects: set[str] = set()
        for raw_row in reader:
            dataset = dataset_name or str(raw_row["dataset"]).strip()
            subject = str(raw_row["subject"]).strip()
            subject_uid = f"{dataset}::{subject}"
            if subject_uid in seen_subjects:
                raise ValueError(f"Duplicate subject row detected in {path}: {subject_uid}")
            seen_subjects.add(subject_uid)
            rows.append(
                SubjectRow(
                    dataset=dataset,
                    score_name=str(raw_row.get("score_name", "")).strip(),
                    subject=subject,
                    split=str(raw_row.get("split", "")).strip(),
                    context=str(raw_row.get("context", "")).strip(),
                    anxiety=finite_float(raw_row["anxiety"]),
                    n_channels=int(round(finite_float(raw_row.get("n_channels", 0)))),
                    source_count=int(round(finite_float(raw_row.get("source_count", 0)))),
                    features={name: finite_float(raw_row[name]) for name in feature_names},
                )
            )
    if not rows:
        raise ValueError(f"No subject rows found in {path}")
    return rows


def split_subject_rows(
    rows: list[SubjectRow],
    val_fraction: float,
    seed: int,
    min_bins: int = 4,
    max_bins: int = 10,
) -> dict[str, list[SubjectRow]]:
    if len(rows) < 4:
        raise ValueError(f"Need at least 4 subjects for subject-level split, got {len(rows)}")

    ordered = sorted(rows, key=lambda row: (row.anxiety, row.subject))
    n_bins = min(max_bins, max(min_bins, len(ordered) // 8 or 1))
    rng = np.random.default_rng(int(seed))
    train_rows: list[SubjectRow] = []
    val_rows: list[SubjectRow] = []

    for bucket in np.array_split(np.array(ordered, dtype=object), n_bins):
        bucket_rows = list(bucket.tolist())
        if not bucket_rows:
            continue
        rng.shuffle(bucket_rows)
        if len(bucket_rows) == 1:
            target_val = 0
        else:
            target_val = int(round(len(bucket_rows) * float(val_fraction)))
            target_val = max(1, min(len(bucket_rows) - 1, target_val))
        val_rows.extend(bucket_rows[:target_val])
        train_rows.extend(bucket_rows[target_val:])

    if not train_rows or not val_rows:
        raise RuntimeError("Split failed to create non-empty train/val sets.")

    return {
        "train": sorted(train_rows, key=lambda row: row.subject_uid),
        "val": sorted(val_rows, key=lambda row: row.subject_uid),
    }


def split_subject_rows_from_subject_lists(
    rows: list[SubjectRow],
    train_subjects: list[str],
    val_subjects: list[str],
) -> dict[str, list[SubjectRow]]:
    train_subjects = [str(subject) for subject in train_subjects]
    val_subjects = [str(subject) for subject in val_subjects]
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(f"Overlap between provided train/val subject lists: {overlap[:5]}")

    row_by_subject = {row.subject: row for row in rows}
    missing_train = sorted(train_set - set(row_by_subject))
    missing_val = sorted(val_set - set(row_by_subject))
    if missing_train or missing_val:
        raise ValueError(
            "Provided subject split contains unknown subjects: "
            f"train_missing={missing_train[:5]} val_missing={missing_val[:5]}"
        )

    train_rows = [row_by_subject[subject] for subject in train_subjects]
    val_rows = [row_by_subject[subject] for subject in val_subjects]
    if not train_rows or not val_rows:
        raise ValueError("Provided subject split must keep non-empty train and val sets.")

    return {
        "train": sorted(train_rows, key=lambda row: row.subject_uid),
        "val": sorted(val_rows, key=lambda row: row.subject_uid),
    }


def build_thresholds(train_rows: list[SubjectRow], gray_z: float) -> DatasetThresholds:
    scores = np.array([row.anxiety for row in train_rows], dtype=np.float32)
    center = float(np.median(scores))
    std = float(np.std(scores))
    if not math.isfinite(std) or std < EPS:
        std = 1.0
    low = float(center - float(gray_z) * std)
    high = float(center + float(gray_z) * std)
    ref = train_rows[0]
    return DatasetThresholds(
        dataset=ref.dataset,
        score_name=ref.score_name,
        center=center,
        std=std,
        gray_z=float(gray_z),
        low_threshold=low,
        high_threshold=high,
    )


def build_thresholds_from_rows(rows: list[SubjectRow], gray_z: float) -> DatasetThresholds:
    scores = np.array([row.anxiety for row in rows], dtype=np.float32)
    center = float(np.median(scores))
    std = float(np.std(scores))
    if not math.isfinite(std) or std < EPS:
        std = 1.0
    low = float(center - float(gray_z) * std)
    high = float(center + float(gray_z) * std)
    ref = rows[0]
    return DatasetThresholds(
        dataset=ref.dataset,
        score_name=ref.score_name,
        center=center,
        std=std,
        gray_z=float(gray_z),
        low_threshold=low,
        high_threshold=high,
    )


def resolve_feature_preset(
    feature_preset: str,
    include_exploratory_features: bool,
) -> tuple[list[str], list[str], list[str]]:
    preset = str(feature_preset).strip().lower()
    if preset == "full":
        input_features = list(DEFAULT_GLOBAL_INPUT_FEATURES + DEFAULT_REGION_INPUT_FEATURES)
        if include_exploratory_features:
            input_features.extend(DEFAULT_EXPLORATORY_FEATURES)
        global_target_features = list(DEFAULT_GLOBAL_CONSTRAINT_FEATURES)
        region_target_features = list(DEFAULT_REGION_CONSTRAINT_FEATURES)
        return input_features, global_target_features, region_target_features
    if preset in {"shared_common", "shared_subspace", "common"}:
        return (
            list(SHARED_COMMON_INPUT_FEATURES),
            list(SHARED_COMMON_GLOBAL_CONSTRAINT_FEATURES),
            list(SHARED_COMMON_REGION_CONSTRAINT_FEATURES),
        )
    raise ValueError(f"Unsupported feature_preset={feature_preset!r}")


def build_normalizer(train_rows: list[SubjectRow], feature_names: list[str]) -> DatasetNormalizer:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature_name in feature_names:
        values = np.array([row.features[feature_name] for row in train_rows], dtype=np.float32)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std < EPS:
            std = 1.0
        means[feature_name] = mean
        stds[feature_name] = std
    return DatasetNormalizer(
        dataset=train_rows[0].dataset,
        feature_means=means,
        feature_stds=stds,
    )


def zscore_feature(value: float, feature_name: str, normalizer: DatasetNormalizer) -> float:
    mean = normalizer.feature_means[feature_name]
    std = normalizer.feature_stds[feature_name]
    return float((float(value) - mean) / (std + EPS))


def build_joint_sample(
    row: SubjectRow,
    dataset_index: int,
    thresholds: DatasetThresholds,
    normalizer: DatasetNormalizer,
    input_features: list[str],
    global_target_features: list[str],
    region_target_features: list[str],
    gray_zone_weight: float,
) -> JointSample:
    label = 1 if float(row.anxiety) >= float(thresholds.center) else 0
    in_gray_zone = bool(thresholds.low_threshold < float(row.anxiety) < thresholds.high_threshold)
    sample_weight = float(gray_zone_weight) if in_gray_zone else 1.0

    x = np.array(
        [zscore_feature(row.features[name], name, normalizer) for name in input_features],
        dtype=np.float32,
    )
    global_target = np.array(
        [zscore_feature(row.features[name], name, normalizer) for name in global_target_features],
        dtype=np.float32,
    )
    region_target = np.array(
        [zscore_feature(row.features[name], name, normalizer) for name in region_target_features],
        dtype=np.float32,
    )
    return JointSample(
        dataset=row.dataset,
        dataset_index=int(dataset_index),
        subject=row.subject,
        subject_uid=row.subject_uid,
        source_split=row.split,
        score_name=row.score_name,
        context=row.context,
        anxiety=float(row.anxiety),
        label=int(label),
        sample_weight=float(sample_weight),
        in_gray_zone=in_gray_zone,
        x=x,
        global_target=global_target,
        region_target=region_target,
    )


def validate_required_features(rows_by_dataset: dict[str, list[SubjectRow]], required_features: list[str]) -> None:
    missing = {}
    for dataset_name, rows in rows_by_dataset.items():
        available = set(rows[0].features.keys())
        missing_here = [name for name in required_features if name not in available]
        if missing_here:
            missing[dataset_name] = missing_here
    if missing:
        raise ValueError(f"Missing required features: {json.dumps(missing, ensure_ascii=False)}")


def build_joint_datasets(
    dataset_csvs: dict[str, Path] | None = None,
    seed: int = 42,
    val_fraction: float = 0.30,
    gray_z: float = 0.35,
    gray_zone_weight: float = 0.35,
    threshold_mode: str = "train_median",
    feature_preset: str = "full",
    include_exploratory_features: bool = False,
    subject_splits_by_dataset: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[JointConstraintDataset, JointConstraintDataset, dict]:
    dataset_csvs = dict(dataset_csvs or DEFAULT_DATASET_CSVS)
    dataset_names = list(dataset_csvs.keys())
    input_features, global_target_features, region_target_features = resolve_feature_preset(
        feature_preset=feature_preset,
        include_exploratory_features=include_exploratory_features,
    )
    required_features = sorted(set(input_features + global_target_features + region_target_features))

    rows_by_dataset = {
        dataset_name: read_subject_feature_csv(path, dataset_name=dataset_name)
        for dataset_name, path in dataset_csvs.items()
    }
    validate_required_features(rows_by_dataset, required_features)

    train_samples: list[JointSample] = []
    val_samples: list[JointSample] = []
    split_info: dict[str, object] = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "gray_z": float(gray_z),
        "gray_zone_weight": float(gray_zone_weight),
        "threshold_mode": str(threshold_mode),
        "feature_preset": str(feature_preset),
        "dataset_order": dataset_names,
        "input_features": input_features,
        "global_constraint_features": global_target_features,
        "region_constraint_features": region_target_features,
        "per_dataset": {},
    }

    for dataset_index, dataset_name in enumerate(dataset_names):
        rows = rows_by_dataset[dataset_name]
        provided_split = (subject_splits_by_dataset or {}).get(dataset_name)
        if provided_split is not None:
            split = split_subject_rows_from_subject_lists(
                rows,
                train_subjects=provided_split["train"],
                val_subjects=provided_split["val"],
            )
            split_source = "provided_subject_lists"
        else:
            split = split_subject_rows(rows, val_fraction=val_fraction, seed=seed + dataset_index * 997)
            split_source = "random_seeded_split"
        threshold_mode_value = str(threshold_mode).strip().lower()
        if threshold_mode_value == "train_median":
            thresholds = build_thresholds(split["train"], gray_z=gray_z)
        elif threshold_mode_value in {"dataset_global_median", "global_dataset_median"}:
            thresholds = build_thresholds_from_rows(rows, gray_z=gray_z)
        else:
            raise ValueError(f"Unsupported threshold_mode={threshold_mode!r}")
        normalizer = build_normalizer(split["train"], feature_names=required_features)

        train_dataset_samples = [
            build_joint_sample(
                row=row,
                dataset_index=dataset_index,
                thresholds=thresholds,
                normalizer=normalizer,
                input_features=input_features,
                global_target_features=global_target_features,
                region_target_features=region_target_features,
                gray_zone_weight=gray_zone_weight,
            )
            for row in split["train"]
        ]
        val_dataset_samples = [
            build_joint_sample(
                row=row,
                dataset_index=dataset_index,
                thresholds=thresholds,
                normalizer=normalizer,
                input_features=input_features,
                global_target_features=global_target_features,
                region_target_features=region_target_features,
                gray_zone_weight=gray_zone_weight,
            )
            for row in split["val"]
        ]
        train_samples.extend(train_dataset_samples)
        val_samples.extend(val_dataset_samples)

        split_info["per_dataset"][dataset_name] = {
            "csv_path": str(dataset_csvs[dataset_name]),
            "score_name": rows[0].score_name,
            "split_source": split_source,
            "threshold_source": threshold_mode_value,
            "train_subjects": [row.subject for row in split["train"]],
            "val_subjects": [row.subject for row in split["val"]],
            "train_count": int(len(split["train"])),
            "val_count": int(len(split["val"])),
            "thresholds": asdict(thresholds),
            "label_counts": {
                "train": {
                    "low": int(sum(sample.label == 0 for sample in train_dataset_samples)),
                    "high": int(sum(sample.label == 1 for sample in train_dataset_samples)),
                    "gray": int(sum(sample.in_gray_zone for sample in train_dataset_samples)),
                },
                "val": {
                    "low": int(sum(sample.label == 0 for sample in val_dataset_samples)),
                    "high": int(sum(sample.label == 1 for sample in val_dataset_samples)),
                    "gray": int(sum(sample.in_gray_zone for sample in val_dataset_samples)),
                },
            },
            "normalizer": asdict(normalizer),
        }

    train_ds = JointConstraintDataset(train_samples)
    val_ds = JointConstraintDataset(val_samples)
    split_info["train_subjects"] = int(len(train_samples))
    split_info["val_subjects"] = int(len(val_samples))
    return train_ds, val_ds, split_info


def weighted_binary_counts(dataset: JointConstraintDataset) -> tuple[float, float]:
    pos = 0.0
    neg = 0.0
    for sample in dataset.samples:
        if sample.label == 1:
            pos += float(sample.sample_weight)
        else:
            neg += float(sample.sample_weight)
    return neg, pos
