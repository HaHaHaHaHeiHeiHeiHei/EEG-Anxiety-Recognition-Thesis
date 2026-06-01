#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文说明

用途：
    通过 OpenNeuro/EEGDash 下载公开 EEG 数据集，并按论文统一流程提取
    subject-level 全局频谱组织特征。
输入：
    OpenNeuro/EEGDash 数据集编号、缓存目录和标签列配置。
输出：
    `subject_features.csv`、`dataset_info.json`、scorecard 和失败记录。
快速运行：
    `python -m anxiety_eeg.features.score_ds007609 --output-dir features/subject_features/ds007609`
论文对应：
    第 3 章公开数据集构建和第 4 章全局频谱特征。
注意事项：
    首次运行会下载公开 EEG 文件；请遵守对应数据集许可和引用要求。
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import shutil
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mne_bids import find_matching_paths, read_raw_bids


from anxiety_eeg.features.common_dataset_eval import (  # noqa: E402
    SubjectFeatureBundle,
    add_common_cli_arguments,
    compute_windowed_channel_features,
    default_scoring_dir,
    extract_channel_positions,
    save_dataset_outputs,
    select_eeg_channels,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "_cache" / "eegdash"
SUBJECT_KEYS = (
    "participant_id",
    "subject",
    "subject_id",
    "participant",
    "sub",
    "id",
)
POSITIVE_LABEL_KEYWORDS = (
    "anxiety",
    "anx",
    "stai",
    "hads_a",
    "hads-anxiety",
    "trait",
    "state",
    "bai",
    "hama",
    "gad",
)
NEGATIVE_LABEL_KEYWORDS = (
    "depress",
    "bdi",
    "stress",
    "anger",
    "sleep",
    "wander",
    "mind",
    "fatigue",
)
TASK_KEYS = (
    "task",
    "task_name",
    "session",
    "condition",
    "run",
    "split",
    "file",
    "path",
    "filename",
)


@dataclass(frozen=True)
class EEGDashDatasetSpec:
    dataset_name: str
    eegdash_dataset_id: str
    score_name_hint: str
    preferred_label_columns: tuple[str, ...]
    openneuro_dataset_id: str | None = None
    task_include_keywords: tuple[str, ...] = ()
    task_exclude_keywords: tuple[str, ...] = ()
    target_sfreq: float | None = 250.0
    max_recordings_per_subject: int | None = None
    notes: str = ""


def load_eegdash_class():
    try:
        from eegdash import EEGDashDataset
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Missing dependency: eegdash. Install it in your EEG environment first, for example:\n"
            "  pip install eegdash\n"
            f"Import error: {exc}"
        )
    return EEGDashDataset


OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
OPENNEURO_USER_AGENT = "graduation_thesis_anxiety_eeg/remote_eval"
OPENNEURO_METADATA_PATTERNS = (
    "dataset_description.json",
    "participants.tsv",
    "participants.csv",
    "participants.json",
    "phenotype/**",
    "task-*.json",
    "README*",
)
OPENNEURO_EEG_EXTENSIONS = (".edf", ".bdf", ".vhdr", ".set", ".fif")


def should_try_openneuro(spec: EEGDashDatasetSpec) -> bool:
    dataset_id = (spec.openneuro_dataset_id or spec.eegdash_dataset_id).lower()
    return dataset_id.startswith("ds")


def post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": OPENNEURO_USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
        text = response.read().decode("utf-8")
    data = json.loads(text)
    if "errors" in data:
        raise RuntimeError(f"OpenNeuro GraphQL error: {data['errors']}")
    return data


def query_openneuro_index(dataset_id: str) -> list[dict[str, Any]]:
    query = {
        "query": (
            "query DatasetFiles($datasetId: ID!) { "
            "dataset(id: $datasetId) { latestSnapshot { id files(recursive: true) { filename urls size id } } } }"
        ),
        "variables": {"datasetId": dataset_id},
    }
    data = post_json(OPENNEURO_GRAPHQL_URL, query, timeout=120.0)
    dataset = data.get("data", {}).get("dataset")
    if not dataset or not dataset.get("latestSnapshot"):
        raise RuntimeError(f"OpenNeuro dataset not found or has no snapshot: {dataset_id}")
    files = dataset["latestSnapshot"].get("files") or []
    if not files:
        raise RuntimeError(f"OpenNeuro dataset has no files: {dataset_id}")
    return files


def matches_any_pattern(path_text: str, patterns: list[str] | tuple[str, ...]) -> bool:
    normalized = path_text.replace("\\", "/")
    for pattern in patterns:
        pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatch(normalized, pattern):
            return True
        if "/" not in pattern and normalized.startswith(pattern.rstrip("*")):
            return True
    return False


def is_hidden_resource_path(path_text: str) -> bool:
    parts = path_text.replace("\\", "/").split("/")
    return any(part.startswith("._") for part in parts)


def download_url_to_path(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": OPENNEURO_USER_AGENT})
    with urllib.request.urlopen(request, timeout=300.0, context=ssl.create_default_context()) as response:
        with out_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def filter_openneuro_files(
    file_index: list[dict[str, Any]],
    include_patterns: list[str],
    *,
    include_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for file_info in file_index:
        remote_path = str(file_info.get("filename", "")).lstrip("/")
        if not remote_path:
            continue
        if is_hidden_resource_path(remote_path):
            continue
        if not matches_any_pattern(remote_path, include_patterns):
            continue
        normalized = remote_path.replace("\\", "/").lower()
        filename = normalized.rsplit("/", 1)[-1]
        if include_keywords is not None or exclude_keywords is not None:
            if "/eeg/" in normalized and "_task-" in filename:
                if not recording_matches_keywords(
                    normalized,
                    include_keywords=list(include_keywords or []),
                    exclude_keywords=list(exclude_keywords or []),
                ):
                    continue
        urls = file_info.get("urls") or []
        if not urls:
            continue
        selected.append(file_info)
    return selected


def ensure_openneuro_files(
    dataset_id: str,
    target_dir: Path,
    include_patterns: list[str],
    *,
    file_index: list[dict[str, Any]] | None = None,
    include_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    log_prefix: str | None = None,
) -> list[str]:
    if file_index is None:
        file_index = query_openneuro_index(dataset_id)
    selected = filter_openneuro_files(
        file_index,
        include_patterns,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
    )
    cached = 0
    for file_info in selected:
        remote_path = str(file_info.get("filename", "")).lstrip("/")
        local_path = target_dir / remote_path
        if local_path.exists():
            cached += 1
    if log_prefix is not None:
        print(
            f"{log_prefix} matched_files={len(selected)} "
            f"cached={cached} need_download={len(selected) - cached}"
        )

    downloaded = []
    for index, file_info in enumerate(selected, start=1):
        remote_path = str(file_info.get("filename", "")).lstrip("/")
        local_path = target_dir / remote_path
        if not local_path.exists():
            if log_prefix is not None:
                print(f"{log_prefix} fetch {index}/{len(selected)} -> {remote_path}")
            download_url_to_path((file_info.get("urls") or [None])[0], local_path)
        downloaded.append(remote_path)
    return downloaded


def bids_record_context(bids_path) -> str:
    pieces = []
    for key in ("subject", "session", "task", "run", "acquisition", "recording", "description"):
        value = getattr(bids_path, key, None)
        if value not in (None, ""):
            pieces.append(f"{key}:{value}")
    if getattr(bids_path, "fpath", None) is not None:
        pieces.append(str(bids_path.fpath))
    return " | ".join(pieces).lower()


def add_remote_cli_arguments(
    parser: argparse.ArgumentParser,
    spec: EEGDashDatasetSpec,
) -> argparse.ArgumentParser:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_scoring_dir(spec.dataset_name))
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional exact label column override, e.g. STAI or hads_anxiety.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help="Optional TSV/CSV file path override for label discovery.",
    )
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument(
        "--max-recordings-per-subject",
        type=int,
        default=spec.max_recordings_per_subject,
        help="Limit recordings aggregated per subject after task filtering.",
    )
    parser.add_argument(
        "--target-sfreq",
        type=float,
        default=(0.0 if spec.target_sfreq is None else float(spec.target_sfreq)),
        help="Resample each recording to this rate before feature extraction. Use 0 to keep native sfreq.",
    )
    parser.add_argument(
        "--task-include-keywords",
        nargs="*",
        default=list(spec.task_include_keywords),
        help="Keep recordings whose metadata text contains any of these keywords.",
    )
    parser.add_argument(
        "--task-exclude-keywords",
        nargs="*",
        default=list(spec.task_exclude_keywords),
        help="Drop recordings whose metadata text contains any of these keywords.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only inspect label/task metadata discovery without loading EEG recordings.",
    )
    add_common_cli_arguments(parser)
    return parser


def canonical_subject_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    for prefix in ("sub-", "subject-", "participant-"):
        if lower.startswith(prefix):
            lower = lower[len(prefix) :]
            break
    lower = lower.strip()
    if not lower:
        return None
    if lower.isdigit():
        return lower.lstrip("0") or "0"
    return lower


def path_from_dataset(dataset, fallback_root: Path, dataset_id: str) -> Path:
    data_dir = getattr(dataset, "data_dir", None)
    if data_dir is not None:
        return Path(data_dir)
    return fallback_root / dataset_id


def ensure_metadata_materialized(dataset, data_dir: Path) -> None:
    if (data_dir / "participants.tsv").exists():
        return
    if (data_dir / "phenotype").exists():
        return
    datasets = getattr(dataset, "datasets", None)
    first = None
    if datasets:
        first = datasets[0]
    else:
        try:
            first = next(iter(dataset))
        except Exception:
            first = None
    if first is None:
        return
    try:
        _ = get_record_raw(first)
    except Exception:
        return


def load_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def detect_subject_column(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    fieldnames = list(rows[0].keys())
    lower_map = {name.lower(): name for name in fieldnames}
    for key in SUBJECT_KEYS:
        if key in lower_map:
            return lower_map[key]
    fallback = [name for name in fieldnames if any(token in name.lower() for token in ("subject", "participant"))]
    return fallback[0] if fallback else None


def score_label_column(
    column_name: str,
    preferred_columns: tuple[str, ...],
    positive_keywords: tuple[str, ...],
    negative_keywords: tuple[str, ...],
    description_text: str = "",
) -> float:
    name = column_name.lower()
    score = 0.0
    seen_preferred = []
    seen_keys = set()
    for preferred in preferred_columns:
        preferred_lower = preferred.lower()
        if preferred_lower in seen_keys:
            continue
        seen_keys.add(preferred_lower)
        seen_preferred.append(preferred_lower)

    for idx, preferred_lower in enumerate(seen_preferred):
        if name == preferred_lower:
            score += 1000.0 - idx
        elif preferred_lower in name:
            score += 180.0 - idx
    text = f"{name} {description_text.lower()}"
    for keyword in positive_keywords:
        if keyword in text:
            score += 40.0
    for keyword in negative_keywords:
        if keyword in text:
            score -= 35.0
    return score


def candidate_metadata_files(data_dir: Path, explicit_file: Path | None = None) -> list[Path]:
    files: list[Path] = []
    if explicit_file is not None:
        if explicit_file.is_absolute():
            files.append(explicit_file)
        else:
            files.append((data_dir / explicit_file).resolve())
        return files

    for name in ("participants.tsv", "participants.csv", "participants.json"):
        path = data_dir / name
        if path.exists():
            files.append(path)

    phenotype_dir = data_dir / "phenotype"
    if phenotype_dir.exists():
        for suffix in ("*.tsv", "*.csv", "*.json"):
            files.extend(sorted(phenotype_dir.glob(suffix)))
    return files


def discover_label_values(
    data_dir: Path,
    preferred_columns: tuple[str, ...],
    positive_keywords: tuple[str, ...],
    negative_keywords: tuple[str, ...],
    explicit_label_column: str | None = None,
    explicit_label_file: Path | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    best_candidate = None
    available_columns: list[dict[str, Any]] = []

    for file_path in candidate_metadata_files(data_dir, explicit_label_file):
        if not file_path.exists():
            continue
        if file_path.suffix.lower() == ".json":
            continue

        json_sidecar = file_path.with_suffix(".json")
        sidecar = {}
        if json_sidecar.exists():
            try:
                sidecar = json.loads(json_sidecar.read_text(encoding="utf-8"))
            except Exception:
                sidecar = {}

        rows = load_table(file_path)
        subject_col = detect_subject_column(rows)
        if subject_col is None:
            continue

        numeric_columns = []
        if rows:
            for column in rows[0].keys():
                if column == subject_col:
                    continue
                values = [numeric_value(row.get(column)) for row in rows]
                good = [value for value in values if value is not None]
                if len(good) >= 3:
                    numeric_columns.append(column)

        for column in numeric_columns:
            if explicit_label_column is not None and column != explicit_label_column:
                continue
            desc = ""
            if isinstance(sidecar, dict):
                col_meta = sidecar.get(column)
                if isinstance(col_meta, dict):
                    desc = " ".join(str(col_meta.get(key, "")) for key in ("LongName", "Description"))

            score = score_label_column(
                column_name=column,
                preferred_columns=preferred_columns,
                positive_keywords=positive_keywords,
                negative_keywords=negative_keywords,
                description_text=desc,
            )
            mapping: dict[str, list[float]] = defaultdict(list)
            aliases: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                raw_subject = row.get(subject_col)
                subject_id = canonical_subject_id(raw_subject)
                value = numeric_value(row.get(column))
                if subject_id is None or value is None:
                    continue
                mapping[subject_id].append(value)
                aliases[subject_id].add(str(raw_subject))
            if not mapping:
                continue

            final_mapping = {subject_id: float(np.mean(values)) for subject_id, values in mapping.items()}
            available_columns.append(
                {
                    "file": str(file_path),
                    "subject_column": subject_col,
                    "label_column": column,
                    "score": score,
                    "n_subjects": len(final_mapping),
                    "subject_alias_example": dict(list((k, sorted(v)) for k, v in aliases.items())[:3]),
                }
            )
            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "file": file_path,
                    "subject_column": subject_col,
                    "label_column": column,
                    "score": score,
                    "mapping": final_mapping,
                    "subject_aliases": {subject_id: sorted(values) for subject_id, values in aliases.items()},
                }

    if best_candidate is None:
        raise ValueError(
            "No anxiety-like numeric label column found. "
            f"Searched under {data_dir}. Available candidates: {available_columns[:20]}"
        )
    if explicit_label_column is None and best_candidate["score"] < 100:
        raise ValueError(
            "Found numeric metadata columns, but none looked anxiety-related enough. "
            f"Top candidate={best_candidate['label_column']} from {best_candidate['file']}. "
            "Pass --label-column to override."
        )

    return best_candidate["mapping"], {
        "label_file": str(best_candidate["file"]),
        "label_column": best_candidate["label_column"],
        "subject_column": best_candidate["subject_column"],
        "score": float(best_candidate["score"]),
        "subject_aliases": best_candidate.get("subject_aliases", {}),
        "available_candidates": sorted(available_columns, key=lambda row: -row["score"])[:30],
    }


def object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            out = value.to_dict()
            if isinstance(out, dict):
                return dict(out)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {key: val for key, val in vars(value).items() if not key.startswith("_")}
        except Exception:
            return {}
    return {}


def extract_record_metadata(record: Any) -> dict[str, Any]:
    meta = {}
    for attr in ("description", "metadata", "record", "info"):
        meta.update(object_to_dict(getattr(record, attr, None)))
    for key in ("subject", "session", "task", "run"):
        value = getattr(record, key, None)
        if value is not None and key not in meta:
            meta[key] = value
    return meta


def record_subject(record: Any, metadata: dict[str, Any]) -> str | None:
    for key in SUBJECT_KEYS:
        if key in metadata:
            return canonical_subject_id(metadata[key])
    for attr in ("subject", "participant_id"):
        value = getattr(record, attr, None)
        subject = canonical_subject_id(value)
        if subject is not None:
            return subject
    return None


def record_context_text(record: Any, metadata: dict[str, Any]) -> str:
    pieces = []
    for key, value in metadata.items():
        key_lower = str(key).lower()
        if any(token in key_lower for token in TASK_KEYS):
            pieces.append(f"{key_lower}:{value}")
    for attr in ("path", "filename", "name"):
        value = getattr(record, attr, None)
        if value is not None:
            pieces.append(str(value))
    return " | ".join(pieces).lower()


def recording_matches_keywords(
    text: str,
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> bool:
    lower = text.lower()
    if include_keywords:
        if not any(keyword.lower() in lower for keyword in include_keywords):
            return False
    if exclude_keywords and any(keyword.lower() in lower for keyword in exclude_keywords):
        return False
    return True


def get_record_raw(record: Any):
    raw = getattr(record, "raw", None)
    if raw is not None:
        return raw
    load_fn = getattr(record, "load", None)
    if callable(load_fn):
        loaded = load_fn()
        raw = getattr(loaded, "raw", None)
        if raw is not None:
            return raw
        return loaded
    raise ValueError(f"Cannot access raw recording for object of type {type(record)!r}")


def maybe_resample_raw(raw, target_sfreq: float | None):
    if target_sfreq is None or target_sfreq <= 0:
        return raw
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - target_sfreq) < 1e-6:
        return raw
    loaded = raw.copy().load_data()
    loaded.resample(target_sfreq, npad="auto")
    return loaded


def dataset_iterable(dataset: Any):
    datasets = getattr(dataset, "datasets", None)
    if datasets is not None:
        return list(datasets)
    return list(dataset)


def collect_eegdash_bundles(
    args: argparse.Namespace,
    spec: EEGDashDatasetSpec,
) -> tuple[list[SubjectFeatureBundle], list[dict[str, Any]], dict[str, Any]]:
    EEGDashDataset = load_eegdash_class()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = EEGDashDataset(dataset=spec.eegdash_dataset_id, cache_dir=str(cache_dir))
    except AssertionError as exc:
        if "empty iterable" in str(exc):
            raise RuntimeError(
                f"EEGDash returned zero records for dataset={spec.eegdash_dataset_id!r}. "
                "This usually means your installed EEGDash registry/database does not expose this dataset yet."
            ) from exc
        raise
    data_dir = path_from_dataset(dataset, cache_dir, spec.eegdash_dataset_id)
    ensure_metadata_materialized(dataset, data_dir)

    label_values, label_info = discover_label_values(
        data_dir=data_dir,
        preferred_columns=spec.preferred_label_columns,
        positive_keywords=POSITIVE_LABEL_KEYWORDS,
        negative_keywords=NEGATIVE_LABEL_KEYWORDS,
        explicit_label_column=args.label_column,
        explicit_label_file=args.label_file,
    )

    bundles: list[SubjectFeatureBundle] = []
    failures: list[dict[str, Any]] = []
    subject_recordings: dict[str, list[tuple[Any, dict[str, Any], str]]] = defaultdict(list)
    task_counter = Counter()
    unmatched_subjects = set(label_values)

    for record in dataset_iterable(dataset):
        metadata = extract_record_metadata(record)
        subject_id = record_subject(record, metadata)
        if subject_id is None or subject_id not in label_values:
            continue
        context_text = record_context_text(record, metadata)
        if not recording_matches_keywords(
            context_text,
            include_keywords=list(args.task_include_keywords),
            exclude_keywords=list(args.task_exclude_keywords),
        ):
            continue
        subject_recordings[subject_id].append((record, metadata, context_text))
        unmatched_subjects.discard(subject_id)
        task_counter[context_text or "unknown"] += 1

    subject_ids = sorted(subject_recordings)
    if args.limit_subjects is not None:
        subject_ids = subject_ids[: args.limit_subjects]

    metadata_summary = {
        "data_dir": str(data_dir),
        "label_info": label_info,
        "n_label_subjects": len(label_values),
        "n_subjects_with_matching_recordings": len(subject_recordings),
        "subjects_without_matching_recordings": sorted(unmatched_subjects)[:50],
        "matched_task_examples": list(task_counter.keys())[:30],
    }

    if args.metadata_only:
        return bundles, failures, metadata_summary

    print(
        f"[{spec.dataset_name}] dataset_id={spec.eegdash_dataset_id} "
        f"subjects_with_labels={len(label_values)} matched_subjects={len(subject_ids)}"
    )
    print(
        f"[{spec.dataset_name}] label_column={label_info['label_column']} "
        f"label_file={label_info['label_file']}"
    )

    for index, subject_id in enumerate(subject_ids, start=1):
        recordings = subject_recordings[subject_id]
        if args.max_recordings_per_subject is not None:
            recordings = recordings[: args.max_recordings_per_subject]

        accumulator: dict[tuple[str, str], list[float]] = defaultdict(list)
        channel_names = set()
        channel_positions: dict[str, list[float]] = {}
        good_recordings = 0
        bad_recordings = 0
        contexts = []

        for record, metadata, context_text in recordings:
            try:
                raw = get_record_raw(record)
                raw = maybe_resample_raw(raw, None if args.target_sfreq <= 0 else float(args.target_sfreq))
                picks, names = select_eeg_channels(raw, common_local_only=args.use_common_local_channels)
                for name, xyz in extract_channel_positions(raw, picks, names).items():
                    channel_positions.setdefault(name, xyz)
                features = compute_windowed_channel_features(
                    raw=raw,
                    picks=picks,
                    channel_names=names,
                    ranges=[(0, raw.n_times)],
                    window_sec=args.window_sec,
                    max_windows=args.max_windows,
                    nperseg_sec=args.nperseg_sec,
                )
                for channel, feature_dict in features.items():
                    channel_names.add(channel)
                    for feature_name, value in feature_dict.items():
                        accumulator[(channel, feature_name)].append(float(value))
                good_recordings += 1
                if context_text:
                    contexts.append(context_text)
            except Exception as exc:
                bad_recordings += 1
                failures.append(
                    {
                        "subject": subject_id,
                        "reason": repr(exc),
                        "record_context": context_text,
                    }
                )

        if not accumulator:
            failures.append(
                {
                    "subject": subject_id,
                    "reason": "no_valid_recordings_after_loading",
                    "record_count": len(recordings),
                }
            )
            print(f"  [skip] {subject_id}: no valid recordings")
            continue

        merged_features: dict[str, dict[str, float]] = defaultdict(dict)
        for (channel, feature_name), values in accumulator.items():
            merged_features[channel][feature_name] = float(np.mean(values))

        context = (
            " | ".join(sorted(set(contexts))[:3])
            if contexts
            else "all_selected_recordings"
        )
        bundles.append(
            SubjectFeatureBundle(
                dataset_name=spec.dataset_name,
                score_name=label_info["label_column"],
                subject=subject_id,
                split="remote",
                anxiety=float(label_values[subject_id]),
                context=context,
                channel_names=sorted(channel_names),
                channel_features=dict(merged_features),
                source_count=good_recordings,
                metadata={
                    "recordings_ok": good_recordings,
                    "recordings_bad": bad_recordings,
                    "eegdash_dataset_id": spec.eegdash_dataset_id,
                    "channel_positions": channel_positions,
                },
            )
        )
        print(
            f"  [{index:03d}/{len(subject_ids):03d}] {subject_id} "
            f"recordings_ok={good_recordings}/{len(recordings)} bad={bad_recordings}"
        )

    metadata_summary["selected_subject_ids"] = subject_ids
    metadata_summary["successful_subjects"] = len(bundles)
    return bundles, failures, metadata_summary


def openneuro_subject_patterns(subject_aliases: list[str], canonical_subject: str) -> list[str]:
    patterns = set()

    def add_eeg_patterns(subject_token: str) -> None:
        token = str(subject_token).strip()
        if not token:
            return
        lower = token.lower()
        if lower.startswith("sub-"):
            subject_name = token
        else:
            subject_name = f"sub-{token}"
        patterns.add(f"{subject_name}/eeg/**")
        patterns.add(f"{subject_name}/ses-*/eeg/**")

    for alias in subject_aliases:
        alias_text = str(alias).strip()
        if not alias_text:
            continue
        add_eeg_patterns(alias_text)
    add_eeg_patterns(canonical_subject)
    return sorted(patterns)


def openneuro_subject_query_ids(subject_aliases: list[str], canonical_subject: str) -> list[str]:
    values = []
    seen = set()

    def add_subject_value(subject_token: str) -> None:
        token = str(subject_token).strip()
        if not token:
            return
        lower = token.lower()
        if lower.startswith("sub-"):
            token = token[4:]
        token = token.strip()
        if not token or token in seen:
            return
        seen.add(token)
        values.append(token)

    for alias in subject_aliases:
        add_subject_value(alias)
    add_subject_value(canonical_subject)
    return values


def cleanup_openneuro_subject_cache(target_dir: Path, subject_aliases: list[str], canonical_subject: str) -> None:
    removed = set()

    def remove_subject_dir(subject_token: str) -> None:
        token = str(subject_token).strip()
        if not token:
            return
        lower = token.lower()
        if lower.startswith("sub-"):
            subject_name = token
        else:
            subject_name = f"sub-{token}"
        local_dir = (target_dir / subject_name).resolve()
        if local_dir in removed:
            return
        removed.add(local_dir)
        if local_dir.exists() and local_dir.is_dir():
            shutil.rmtree(local_dir, ignore_errors=True)

    for alias in subject_aliases:
        remove_subject_dir(alias)
    remove_subject_dir(canonical_subject)


def collect_openneuro_bundles(
    args: argparse.Namespace,
    spec: EEGDashDatasetSpec,
) -> tuple[list[SubjectFeatureBundle], list[dict[str, Any]], dict[str, Any]]:
    dataset_id = spec.openneuro_dataset_id or spec.eegdash_dataset_id
    cache_dir = args.cache_dir.resolve()
    data_dir = cache_dir / dataset_id
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{spec.dataset_name}] OpenNeuro indexing dataset={dataset_id} ...")
    file_index = query_openneuro_index(dataset_id)
    print(f"[{spec.dataset_name}] OpenNeuro index ready: total_files={len(file_index)}")

    metadata_downloaded = ensure_openneuro_files(
        dataset_id,
        data_dir,
        list(OPENNEURO_METADATA_PATTERNS),
        file_index=file_index,
        log_prefix=f"[{spec.dataset_name}] [metadata]",
    )
    label_values, label_info = discover_label_values(
        data_dir=data_dir,
        preferred_columns=spec.preferred_label_columns,
        positive_keywords=POSITIVE_LABEL_KEYWORDS,
        negative_keywords=NEGATIVE_LABEL_KEYWORDS,
        explicit_label_column=args.label_column,
        explicit_label_file=args.label_file,
    )

    metadata_summary = {
        "source": "openneuro",
        "openneuro_dataset_id": dataset_id,
        "data_dir": str(data_dir),
        "label_info": label_info,
        "n_label_subjects": len(label_values),
        "downloaded_metadata_files": metadata_downloaded[:200],
    }

    if args.metadata_only:
        return [], [], metadata_summary

    subject_ids = sorted(label_values)
    if args.limit_subjects is not None:
        subject_ids = subject_ids[: args.limit_subjects]

    print(
        f"[{spec.dataset_name}] OpenNeuro subjects planned={len(subject_ids)} "
        f"label_column={label_info['label_column']}"
    )

    base_patterns = ["dataset_description.json", "participants.tsv", "participants.json", "participants.csv"]
    grouped_paths: dict[str, list[Any]] = defaultdict(list)
    task_counter = Counter()
    bundles: list[SubjectFeatureBundle] = []
    failures: list[dict[str, Any]] = []
    selected_subject_ids: list[str] = []

    print(
        f"[{spec.dataset_name}] label_column={label_info['label_column']} "
        f"label_file={label_info['label_file']}"
    )

    for index, subject_id in enumerate(subject_ids, start=1):
        aliases = label_info.get("subject_aliases", {}).get(subject_id, [])
        include_patterns = base_patterns + openneuro_subject_patterns(aliases, subject_id)
        query_subjects = openneuro_subject_query_ids(aliases, subject_id)

        try:
            ensure_openneuro_files(
                dataset_id,
                data_dir,
                include_patterns,
                file_index=file_index,
                include_keywords=list(args.task_include_keywords),
                exclude_keywords=list(args.task_exclude_keywords),
                log_prefix=f"[{spec.dataset_name}] [download {index:03d}/{len(subject_ids):03d}] subject={subject_id}",
            )

            subject_paths = find_matching_paths(
                root=data_dir,
                subjects=query_subjects,
                datatypes="eeg",
                suffixes="eeg",
                extensions=list(OPENNEURO_EEG_EXTENSIONS),
                ignore_json=True,
                check=False,
            )

            filtered_paths = []
            for bids_path in subject_paths:
                matched_subject = canonical_subject_id(getattr(bids_path, "subject", None))
                if matched_subject != subject_id:
                    continue
                context = bids_record_context(bids_path)
                if not recording_matches_keywords(
                    context,
                    include_keywords=list(args.task_include_keywords),
                    exclude_keywords=list(args.task_exclude_keywords),
                ):
                    continue
                filtered_paths.append(bids_path)
                grouped_paths[subject_id].append(bids_path)
                task_counter[context or "unknown"] += 1

            print(
                f"[{spec.dataset_name}] [scan {index:03d}/{len(subject_ids):03d}] "
                f"subject={subject_id} bids_paths={len(subject_paths)} kept={len(filtered_paths)}"
            )

            if not filtered_paths:
                failures.append(
                    {
                        "subject": subject_id,
                        "reason": "no_matching_bids_paths_after_task_filter",
                        "record_count": 0,
                    }
                )
                continue

            selected_subject_ids.append(subject_id)
            if args.max_recordings_per_subject is not None:
                filtered_paths = filtered_paths[: args.max_recordings_per_subject]

            accumulator: dict[tuple[str, str], list[float]] = defaultdict(list)
            channel_names = set()
            channel_positions: dict[str, list[float]] = {}
            contexts = []
            good_recordings = 0
            bad_recordings = 0

            for bids_path in filtered_paths:
                try:
                    raw = read_raw_bids(bids_path=bids_path, verbose="ERROR")
                    raw = maybe_resample_raw(raw, None if args.target_sfreq <= 0 else float(args.target_sfreq))
                    picks, names = select_eeg_channels(raw, common_local_only=args.use_common_local_channels)
                    for name, xyz in extract_channel_positions(raw, picks, names).items():
                        channel_positions.setdefault(name, xyz)
                    features = compute_windowed_channel_features(
                        raw=raw,
                        picks=picks,
                        channel_names=names,
                        ranges=[(0, raw.n_times)],
                        window_sec=args.window_sec,
                        max_windows=args.max_windows,
                        nperseg_sec=args.nperseg_sec,
                    )
                    for channel, feature_dict in features.items():
                        channel_names.add(channel)
                        for feature_name, value in feature_dict.items():
                            accumulator[(channel, feature_name)].append(float(value))
                    good_recordings += 1
                    contexts.append(bids_record_context(bids_path))
                except Exception as exc:
                    bad_recordings += 1
                    failures.append(
                        {
                            "subject": subject_id,
                            "reason": repr(exc),
                            "record_context": bids_record_context(bids_path),
                        }
                    )

            if not accumulator:
                failures.append(
                    {
                        "subject": subject_id,
                        "reason": "no_valid_recordings_after_loading",
                        "record_count": len(filtered_paths),
                    }
                )
                print(f"  [skip] {subject_id}: no valid recordings")
                continue

            merged_features: dict[str, dict[str, float]] = defaultdict(dict)
            for (channel, feature_name), values in accumulator.items():
                merged_features[channel][feature_name] = float(np.mean(values))

            bundles.append(
                SubjectFeatureBundle(
                    dataset_name=spec.dataset_name,
                    score_name=label_info["label_column"],
                    subject=subject_id,
                    split="remote",
                    anxiety=float(label_values[subject_id]),
                    context=" | ".join(sorted(set(contexts))[:3]) if contexts else "all_selected_recordings",
                    channel_names=sorted(channel_names),
                    channel_features=dict(merged_features),
                    source_count=good_recordings,
                    metadata={
                        "recordings_ok": good_recordings,
                        "recordings_bad": bad_recordings,
                        "openneuro_dataset_id": dataset_id,
                        "channel_positions": channel_positions,
                    },
                )
            )
            print(
                f"  [{len(selected_subject_ids):03d}/{len(subject_ids):03d}] {subject_id} "
                f"recordings_ok={good_recordings}/{len(filtered_paths)} bad={bad_recordings}"
            )
        finally:
            cleanup_openneuro_subject_cache(data_dir, aliases, subject_id)

    print(
        f"[{spec.dataset_name}] openneuro_dataset={dataset_id} "
        f"subjects_with_labels={len(label_values)} matched_subjects={len(selected_subject_ids)}"
    )

    metadata_summary["matched_task_examples"] = list(task_counter.keys())[:30]
    metadata_summary["successful_subjects"] = len(bundles)
    metadata_summary["selected_subject_ids"] = selected_subject_ids
    return bundles, failures, metadata_summary


def run_remote_dataset_pipeline(
    args: argparse.Namespace,
    spec: EEGDashDatasetSpec,
) -> int:
    t0 = time.time()
    bundle_source = "eegdash"
    try:
        bundles, failures, metadata_summary = collect_eegdash_bundles(args, spec)
    except Exception as exc:
        if not should_try_openneuro(spec):
            raise
        bundle_source = "openneuro"
        print(f"[{spec.dataset_name}] EEGDash path unavailable: {exc}")
        print(f"[{spec.dataset_name}] Falling back to OpenNeuro direct access...")
        bundles, failures, metadata_summary = collect_openneuro_bundles(args, spec)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata_only:
        path = output_dir / "metadata_probe.json"
        path.write_text(json.dumps(metadata_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{spec.dataset_name}] metadata-only summary -> {path}")
        return 0

    if not bundles:
        (output_dir / "metadata_probe.json").write_text(
            json.dumps(metadata_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(
            f"No valid labeled subjects collected for {spec.dataset_name}. "
            "Run with --metadata-only to inspect task/label discovery."
        )

    config = {
        "eegdash_dataset_id": spec.eegdash_dataset_id,
        "bundle_source": bundle_source,
        "openneuro_dataset_id": spec.openneuro_dataset_id or spec.eegdash_dataset_id,
        "cache_dir": str(args.cache_dir.resolve()),
        "label_column_override": args.label_column,
        "label_file_override": str(args.label_file) if args.label_file is not None else "",
        "task_include_keywords": list(args.task_include_keywords),
        "task_exclude_keywords": list(args.task_exclude_keywords),
        "target_sfreq": None if args.target_sfreq <= 0 else float(args.target_sfreq),
        "window_sec": float(args.window_sec),
        "max_windows": int(args.max_windows),
        "nperseg_sec": float(args.nperseg_sec),
        "min_n": int(args.min_n),
        "use_common_local_channels": bool(args.use_common_local_channels),
        "max_recordings_per_subject": args.max_recordings_per_subject,
        "label_discovery": metadata_summary.get("label_info", {}),
        "notes": spec.notes,
    }

    outputs = save_dataset_outputs(
        dataset_name=spec.dataset_name,
        score_name=metadata_summary["label_info"]["label_column"],
        bundles=bundles,
        failures=failures,
        source_root=Path(metadata_summary["data_dir"]),
        output_dir=output_dir,
        config=config,
        min_n=args.min_n,
    )
    (output_dir / "metadata_probe.json").write_text(
        json.dumps(metadata_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    print(
        f"[{spec.dataset_name}] subjects={len(outputs['subject_rows'])} "
        f"features={len(outputs['feature_scores'])}"
    )
    print(f"[{spec.dataset_name}] output -> {output_dir}")
    print(f"[{spec.dataset_name}] elapsed_sec={elapsed:.1f}")
    return 0


__all__ = [
    "DEFAULT_CACHE_DIR",
    "EEGDashDatasetSpec",
    "add_remote_cli_arguments",
    "run_remote_dataset_pipeline",
]
