"""中文说明

用途：
    训练论文主模型 JointConstraintNet，复现全局频谱组织 + dataset adapter +
    auxiliary constraints 的跨数据集 subject-level 焦虑识别实验。
输入：
    `--features-root` 指向包含 `original_local/ds003478/ds007609` 三个子目录的
    受试者级特征根目录，每个子目录内必须有 `subject_features.csv`。
输出：
    每个 seed 的 checkpoint、history、验证预测、按数据集指标和聚合 summary，
    默认写入 `outputs/joint_constraint`。
快速运行：
    `python scripts/train_joint.py --features-root tests/fixtures/subject_features --seeds 42 --epochs 1 --min-epochs 1 --patience 1 --device cpu`
论文对应：
    第 4 章模型方法与第 5 章主模型/消融实验。
注意事项：
    真实论文结果需要完整特征表和多 seed；smoke 命令只验证代码链路可运行。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from anxiety_eeg.config import apply_json_config
from anxiety_eeg.data.joint_dataset import (
    DEFAULT_FEATURES_ROOT,
    dataset_csvs_from_root,
    build_joint_datasets,
    weighted_binary_counts,
)
from anxiety_eeg.models.joint_constraint import JointConstraintNet


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "joint_constraint"
DEFAULT_SEEDS = [42, 123, 224, 3407, 65422, 7, 21, 2024, 31415, 27182]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint constraint training over original_local, ds003478, and ds007609 subject-level EEG features."
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config file; keys match CLI argument names.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--pretrain-source",
        choices=["none", "ds003478"],
        default="none",
        help="Optional same-feature pretraining stage before three-dataset joint fine-tuning.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--gray-z", type=float, default=0.35)
    parser.add_argument("--gray-zone-weight", type=float, default=0.35)
    parser.add_argument(
        "--threshold-mode",
        choices=["train_median", "dataset_global_median"],
        default="train_median",
        help="Use train-only median thresholds or fixed per-dataset global median thresholds.",
    )
    parser.add_argument(
        "--feature-preset",
        choices=["full", "shared_common"],
        default="full",
        help="Full feature input or shared subspace that is available across all internal/external datasets.",
    )
    parser.add_argument("--include-exploratory-features", action="store_true")
    parser.add_argument("--freeze-adapter", action="store_true", help="Zero and freeze dataset adapter parameters.")
    parser.add_argument("--freeze-dataset-bias", action="store_true", help="Zero and freeze dataset bias parameters.")
    parser.add_argument("--experiment-name", type=str, default="joint_constraint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--pretrain-min-epochs", type=int, default=15)
    parser.add_argument("--pretrain-patience", type=int, default=15)
    parser.add_argument("--pretrain-lr", type=float, default=5e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-dim", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--global-loss-weight", type=float, default=0.35)
    parser.add_argument("--region-loss-weight", type=float, default=0.20)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return apply_json_config(parser.parse_args())


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def is_numeric_scalar(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def finite(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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
    probs: np.ndarray,
    in_gray_zone: np.ndarray,
) -> dict[str, float | list[list[int]]]:
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


def selection_score(metrics: dict[str, float]) -> float:
    return (
        0.35 * max(0.0, finite(metrics.get("extreme_roc_auc")))
        + 0.25 * max(0.0, finite(metrics.get("extreme_balanced_accuracy")))
        + 0.25 * max(0.0, finite(metrics.get("roc_auc")))
        + 0.15 * max(0.0, finite(metrics.get("balanced_accuracy")))
    )


def flatten_metrics(prefix: str, loss: float, metrics: dict[str, float | list[list[int]]]) -> dict[str, float]:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
        "extreme_accuracy",
        "extreme_balanced_accuracy",
        "extreme_f1",
        "extreme_roc_auc",
        "extreme_pr_auc",
    ]
    out = {f"{prefix}_loss": float(loss)}
    for key in keys:
        out[f"{prefix}_{key}"] = metrics.get(key)
    return out


def build_model(args: argparse.Namespace, split_info: dict, device: torch.device) -> JointConstraintNet:
    input_dim = len(split_info["input_features"])
    dataset_count = len(split_info["dataset_order"])
    global_dim = len(split_info["global_constraint_features"])
    region_dim = len(split_info["region_constraint_features"])
    return JointConstraintNet(
        input_dim=input_dim,
        dataset_count=dataset_count,
        global_dim=global_dim,
        region_dim=region_dim,
        hidden_dim=args.hidden_dim,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
    ).to(device)


def zero_and_freeze_module(module) -> None:
    for param in module.parameters():
        param.data.zero_()
        param.requires_grad = False


def apply_model_ablation_settings(model: JointConstraintNet, args: argparse.Namespace) -> None:
    if bool(args.freeze_adapter):
        zero_and_freeze_module(model.dataset_embedding)
        zero_and_freeze_module(model.adapter)
    if bool(args.freeze_dataset_bias):
        zero_and_freeze_module(model.dataset_bias)


def make_loaders(
    train_ds,
    val_ds,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader


def make_optimizer_scheduler(
    model: JointConstraintNet,
    lr: float,
    weight_decay: float,
    epochs: int,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    return optimizer, scheduler


def pos_weight_for_dataset(dataset, device: torch.device) -> tuple[torch.Tensor, float]:
    neg_weight, pos_weight_value = weighted_binary_counts(dataset)
    pos_weight_scalar = float(neg_weight / max(pos_weight_value, 1e-6))
    pos_weight = torch.tensor(pos_weight_scalar, dtype=torch.float32, device=device)
    return pos_weight, pos_weight_scalar


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, best_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "best_metrics": best_metrics,
        },
        path,
    )


def run_epoch(
    model: JointConstraintNet,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    optimizer=None,
    global_loss_weight: float = 0.35,
    region_loss_weight: float = 0.20,
    grad_clip: float = 2.0,
) -> dict:
    train = optimizer is not None
    model.train(train)

    total_loss = []
    total_cls = []
    total_global = []
    total_region = []

    all_labels = []
    all_probs = []
    all_scores = []
    all_gray = []
    all_datasets = []
    all_subjects = []
    all_anxiety = []
    all_weights = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        dataset_index = batch["dataset_index"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        global_target = batch["global_target"].to(device, non_blocking=True)
        region_target = batch["region_target"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(x, dataset_index)
            cls_loss_raw = F.binary_cross_entropy_with_logits(
                out["logit"],
                labels,
                pos_weight=pos_weight,
                reduction="none",
            )
            cls_loss = torch.sum(cls_loss_raw * sample_weight) / torch.clamp(sample_weight.sum(), min=1.0)
            global_loss = F.smooth_l1_loss(out["global_pred"], global_target)
            region_loss = F.smooth_l1_loss(out["region_pred"], region_target)
            loss = cls_loss + float(global_loss_weight) * global_loss + float(region_loss_weight) * region_loss
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    float(grad_clip),
                )
                optimizer.step()

        probs = torch.sigmoid(out["logit"].detach()).cpu().numpy()
        total_loss.append(float(loss.detach().cpu().item()))
        total_cls.append(float(cls_loss.detach().cpu().item()))
        total_global.append(float(global_loss.detach().cpu().item()))
        total_region.append(float(region_loss.detach().cpu().item()))

        all_probs.extend(probs.tolist())
        all_scores.extend(out["logit"].detach().cpu().numpy().tolist())
        all_labels.extend(batch["label"].detach().cpu().numpy().astype(np.int64).tolist())
        all_gray.extend(batch["in_gray_zone"].detach().cpu().numpy().astype(bool).tolist())
        all_datasets.extend(list(batch["dataset"]))
        all_subjects.extend(list(batch["subject"]))
        all_anxiety.extend(batch["anxiety"].detach().cpu().numpy().tolist())
        all_weights.extend(batch["sample_weight"].detach().cpu().numpy().tolist())

    arrays = {
        "labels": np.array(all_labels, dtype=np.int64),
        "probs": np.array(all_probs, dtype=np.float32),
        "scores": np.array(all_scores, dtype=np.float32),
        "in_gray_zone": np.array(all_gray, dtype=bool),
        "dataset": np.array(all_datasets, dtype=object),
        "subject": np.array(all_subjects, dtype=object),
        "anxiety": np.array(all_anxiety, dtype=np.float32),
        "sample_weight": np.array(all_weights, dtype=np.float32),
    }
    metrics = binary_metrics(arrays["labels"], arrays["probs"], arrays["in_gray_zone"])
    return {
        "loss": float(np.mean(total_loss)) if total_loss else float("nan"),
        "cls_loss": float(np.mean(total_cls)) if total_cls else float("nan"),
        "global_loss": float(np.mean(total_global)) if total_global else float("nan"),
        "region_loss": float(np.mean(total_region)) if total_region else float("nan"),
        "arrays": arrays,
        "metrics": metrics,
    }


def metrics_by_dataset(arrays: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for dataset_name in sorted(set(arrays["dataset"].tolist())):
        mask = arrays["dataset"] == dataset_name
        metrics = binary_metrics(arrays["labels"][mask], arrays["probs"][mask], arrays["in_gray_zone"][mask])
        rows.append(
            {
                "dataset": dataset_name,
                "n_subjects": int(np.sum(mask)),
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
            }
        )
    return rows


def prediction_rows(arrays: dict[str, np.ndarray]) -> list[dict]:
    pred = (arrays["probs"] >= 0.5).astype(np.int64)
    rows = []
    for index in range(len(arrays["labels"])):
        rows.append(
            {
                "dataset": str(arrays["dataset"][index]),
                "subject": str(arrays["subject"][index]),
                "label": int(arrays["labels"][index]),
                "pred": int(pred[index]),
                "prob_high": float(arrays["probs"][index]),
                "logit": float(arrays["scores"][index]),
                "anxiety": float(arrays["anxiety"][index]),
                "sample_weight": float(arrays["sample_weight"][index]),
                "in_gray_zone": bool(arrays["in_gray_zone"][index]),
            }
        )
    return rows


def run_training_stage(
    stage_label: str,
    file_prefix: str,
    model: JointConstraintNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds,
    val_ds,
    device: torch.device,
    pos_weight: torch.Tensor,
    pos_weight_scalar: float,
    epochs: int,
    min_epochs: int,
    patience_limit: int,
    lr: float,
    weight_decay: float,
    global_loss_weight: float,
    region_loss_weight: float,
    grad_clip: float,
    save_dir: Path,
    config: dict,
) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    optimizer, scheduler = make_optimizer_scheduler(model, lr=lr, weight_decay=weight_decay, epochs=epochs)
    history_path = save_dir / f"history_{file_prefix}.csv"
    best_path = save_dir / f"best_{file_prefix}.pt"
    summary_path = save_dir / f"summary_{file_prefix}.csv"
    config_path = save_dir / f"config_{file_prefix}.json"
    predictions_path = save_dir / f"val_subject_predictions_{file_prefix}_best.csv"
    dataset_metrics_path = save_dir / f"val_metrics_by_dataset_{file_prefix}_best.csv"

    config = {
        **config,
        "stage_label": stage_label,
        "file_prefix": file_prefix,
        "checkpoint_path": str(best_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_path),
        "dataset_metrics_path": str(dataset_metrics_path),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "stage_lr": float(lr),
        "stage_weight_decay": float(weight_decay),
        "stage_epochs": int(epochs),
        "stage_min_epochs": int(min_epochs),
        "stage_patience": int(patience_limit),
        "pos_weight": float(pos_weight_scalar),
    }
    write_json(config_path, config)

    history = []
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict = {}
    best_dataset_metrics: list[dict] = []
    patience = 0
    started = time.time()

    print(f"\n----- stage={stage_label} -----")
    print(
        f"[Stage Data] train_subjects={len(train_ds)} val_subjects={len(val_ds)} "
        f"pos_weight={pos_weight_scalar:.4f} lr={lr} weight_decay={weight_decay}"
    )

    for epoch in range(1, int(epochs) + 1):
        train_out = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            pos_weight=pos_weight,
            optimizer=optimizer,
            global_loss_weight=global_loss_weight,
            region_loss_weight=region_loss_weight,
            grad_clip=grad_clip,
        )
        val_out = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            pos_weight=pos_weight,
            optimizer=None,
            global_loss_weight=global_loss_weight,
            region_loss_weight=region_loss_weight,
            grad_clip=grad_clip,
        )
        scheduler.step()

        score = selection_score(val_out["metrics"])
        row = {
            "epoch": int(epoch),
            "selection_score": float(score),
            "train_cls_loss": train_out["cls_loss"],
            "train_global_loss": train_out["global_loss"],
            "train_region_loss": train_out["region_loss"],
            "val_cls_loss": val_out["cls_loss"],
            "val_global_loss": val_out["global_loss"],
            "val_region_loss": val_out["region_loss"],
            **flatten_metrics("train", train_out["loss"], train_out["metrics"]),
            **flatten_metrics("val", val_out["loss"], val_out["metrics"]),
        }
        history.append(row)
        write_csv(history_path, history)

        print(
            f"[{stage_label} Epoch {epoch:03d}] "
            f"train_loss={train_out['loss']:.4f} val_loss={val_out['loss']:.4f} "
            f"val_bal={finite(val_out['metrics'].get('balanced_accuracy')):.4f} "
            f"val_auc={finite(val_out['metrics'].get('roc_auc')):.4f} "
            f"val_ext_bal={finite(val_out['metrics'].get('extreme_balanced_accuracy')):.4f} "
            f"val_ext_auc={finite(val_out['metrics'].get('extreme_roc_auc')):.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = {
                "train": train_out["metrics"],
                "val": val_out["metrics"],
                "selection_score": float(score),
            }
            best_dataset_metrics = metrics_by_dataset(val_out["arrays"])
            save_checkpoint(best_path, model, optimizer, epoch, config, best_metrics)
            write_csv(predictions_path, prediction_rows(val_out["arrays"]))
            write_csv(dataset_metrics_path, best_dataset_metrics)
            patience = 0
        else:
            patience += 1

        if epoch >= int(min_epochs) and patience >= int(patience_limit):
            print(f"[{stage_label}] early stop after {patience_limit} stale epochs.")
            break

    elapsed = time.time() - started
    stage_summary = {
        "stage": stage_label,
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "checkpoint": str(best_path),
        "elapsed_sec": float(elapsed),
        "train_subjects": int(len(train_ds)),
        "val_subjects": int(len(val_ds)),
    }
    for key, value in flatten_metrics("best_val", float("nan"), best_metrics.get("val", {})).items():
        if key != "best_val_loss":
            stage_summary[key] = value
    write_csv(summary_path, [stage_summary])

    config["best_epoch"] = int(best_epoch)
    config["best_selection_score"] = float(best_score)
    write_json(config_path, config)
    print(f"[{stage_label} Best] epoch={best_epoch} score={best_score:.4f} checkpoint={best_path}")
    return {
        "summary": stage_summary,
        "dataset_metrics": best_dataset_metrics,
        "checkpoint": str(best_path),
        "config": config,
    }


def load_pretrained_into_finetune(
    pretrained_checkpoint_path: Path,
    finetune_model: JointConstraintNet,
    source_dataset_name: str,
    finetune_dataset_order: list[str],
) -> dict:
    try:
        checkpoint = torch.load(str(pretrained_checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(pretrained_checkpoint_path), map_location="cpu")
    pretrained_state = checkpoint["model_state_dict"]
    target_state = finetune_model.state_dict()

    copied_shared = []
    skipped = []
    for key, value in pretrained_state.items():
        if key in {"dataset_embedding.weight", "dataset_bias.weight"}:
            continue
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value.clone()
            copied_shared.append(key)
        else:
            skipped.append(key)

    adapter_report = {"dataset_embedding": False, "dataset_bias": False}
    target_dataset_index = finetune_dataset_order.index(source_dataset_name)
    if "dataset_embedding.weight" in pretrained_state:
        src_embed = pretrained_state["dataset_embedding.weight"]
        tgt_embed = target_state["dataset_embedding.weight"]
        if src_embed.ndim == 2 and src_embed.shape[0] >= 1 and src_embed.shape[1] == tgt_embed.shape[1]:
            tgt_embed[target_dataset_index] = src_embed[0].clone()
            adapter_report["dataset_embedding"] = True
    if "dataset_bias.weight" in pretrained_state:
        src_bias = pretrained_state["dataset_bias.weight"]
        tgt_bias = target_state["dataset_bias.weight"]
        if src_bias.ndim == 2 and src_bias.shape[0] >= 1 and src_bias.shape[1] == tgt_bias.shape[1]:
            tgt_bias[target_dataset_index] = src_bias[0].clone()
            adapter_report["dataset_bias"] = True

    finetune_model.load_state_dict(target_state)
    return {
        "pretrained_checkpoint": str(pretrained_checkpoint_path),
        "source_dataset": source_dataset_name,
        "target_dataset_index": int(target_dataset_index),
        "copied_shared_keys": int(len(copied_shared)),
        "copied_shared_key_names": copied_shared,
        "adapter_copy": adapter_report,
        "skipped_keys": skipped,
    }


def aggregate_numeric_rows(
    rows: list[dict],
    group_key: str | None,
    exclude_metrics: set[str] | None = None,
) -> list[dict]:
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


def build_aggregate_payload(
    seeds: list[int],
    summary_rows: list[dict],
    dataset_rows: list[dict],
) -> tuple[list[dict], dict]:
    summary_agg_rows = aggregate_numeric_rows(
        summary_rows,
        group_key=None,
        exclude_metrics={"seed"},
    )
    dataset_agg_rows = aggregate_numeric_rows(
        dataset_rows,
        group_key="dataset",
        exclude_metrics={"seed"},
    )
    payload = {
        "n_seeds": int(len(seeds)),
        "seeds": [int(seed) for seed in seeds],
        "overall": {
            row["metric"]: {
                "n": row["n"],
                "mean": row["mean"],
                "std": row["std"],
                "min": row["min"],
                "median": row["median"],
                "max": row["max"],
            }
            for row in summary_agg_rows
        },
        "by_dataset": {},
    }
    for row in dataset_agg_rows:
        dataset_name = row["group"]
        payload["by_dataset"].setdefault(dataset_name, {})
        payload["by_dataset"][dataset_name][row["metric"]] = {
            "n": row["n"],
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "median": row["median"],
            "max": row["max"],
        }
    return summary_agg_rows + dataset_agg_rows, payload


def print_aggregate_report(aggregate_payload: dict) -> None:
    overall = aggregate_payload.get("overall", {})
    print("\n========== aggregate over seeds ==========")
    print(f"[Seeds] n={aggregate_payload.get('n_seeds')} values={aggregate_payload.get('seeds')}")
    for metric_name in (
        "best_selection_score",
        "best_val_balanced_accuracy",
        "best_val_roc_auc",
        "best_val_extreme_balanced_accuracy",
        "best_val_extreme_roc_auc",
    ):
        metric = overall.get(metric_name)
        if not metric:
            continue
        print(
            f"[Overall] {metric_name} "
            f"mean={metric['mean']:.4f} std={metric['std']:.4f} "
            f"median={metric['median']:.4f} range=({metric['min']:.4f}, {metric['max']:.4f})"
        )
    for dataset_name, metrics in sorted(aggregate_payload.get("by_dataset", {}).items()):
        bal = metrics.get("balanced_accuracy")
        auc = metrics.get("roc_auc")
        ext_bal = metrics.get("extreme_balanced_accuracy")
        ext_auc = metrics.get("extreme_roc_auc")
        if bal and auc:
            print(
                f"[Dataset:{dataset_name}] "
                f"bal={bal['mean']:.4f}±{bal['std']:.4f} "
                f"auc={auc['mean']:.4f}±{auc['std']:.4f} "
                f"ext_bal={ext_bal['mean']:.4f}±{ext_bal['std']:.4f} "
                f"ext_auc={ext_auc['mean']:.4f}±{ext_auc['std']:.4f}"
            )


def run_one_seed(args: argparse.Namespace, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    save_dir = Path(args.output_root) / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_csvs = dataset_csvs_from_root(args.features_root)

    finetune_train_ds, finetune_val_ds, split_info = build_joint_datasets(
        dataset_csvs=dataset_csvs,
        seed=seed,
        val_fraction=args.val_fraction,
        gray_z=args.gray_z,
        gray_zone_weight=args.gray_zone_weight,
        threshold_mode=args.threshold_mode,
        feature_preset=args.feature_preset,
        include_exploratory_features=args.include_exploratory_features,
    )

    print(f"\n========== joint_constraint seed={seed} ==========")
    print(
        f"[Data] train_subjects={len(finetune_train_ds)} val_subjects={len(finetune_val_ds)} "
        f"datasets={split_info['dataset_order']}"
    )
    for dataset_name, info in split_info["per_dataset"].items():
        print(
            f"[Split:{dataset_name}] train={info['train_count']} val={info['val_count']} "
            f"labels(train)={info['label_counts']['train']} thresholds={info['thresholds']}"
        )
    print(
        f"[Model] input_dim={len(split_info['input_features'])} global_dim={len(split_info['global_constraint_features'])} "
        f"region_dim={len(split_info['region_constraint_features'])} adapter_dim={args.adapter_dim}"
    )

    pretrain_result = None
    pretrain_transfer = None
    if args.pretrain_source != "none":
        pretrain_subject_splits = {
            args.pretrain_source: {
                "train": list(split_info["per_dataset"][args.pretrain_source]["train_subjects"]),
                "val": list(split_info["per_dataset"][args.pretrain_source]["val_subjects"]),
            }
        }
        pretrain_train_ds, pretrain_val_ds, pretrain_split_info = build_joint_datasets(
            dataset_csvs={args.pretrain_source: dataset_csvs[args.pretrain_source]},
            seed=seed,
            val_fraction=args.val_fraction,
            gray_z=args.gray_z,
            gray_zone_weight=args.gray_zone_weight,
            threshold_mode=args.threshold_mode,
            feature_preset=args.feature_preset,
            include_exploratory_features=args.include_exploratory_features,
            subject_splits_by_dataset=pretrain_subject_splits,
        )
        pretrain_train_loader, pretrain_val_loader = make_loaders(
            pretrain_train_ds,
            pretrain_val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seed,
            device=device,
        )
        pretrain_pos_weight, pretrain_pos_weight_scalar = pos_weight_for_dataset(pretrain_train_ds, device)
        pretrain_model = build_model(args, pretrain_split_info, device)
        apply_model_ablation_settings(pretrain_model, args)
        pretrain_config = {
            "model_family": "JointConstraintNet",
            "training_role": "same_feature_pretrain_before_joint_constraint",
            "source_dataset": args.pretrain_source,
            "seed": int(seed),
            "device": str(device),
            "dataset_csvs": {args.pretrain_source: str(dataset_csvs[args.pretrain_source])},
            "split_rule": "reuse_finetune_subject_split_for_source_dataset",
            "gray_zone_rule": "label by train median, weight gray band by train median +/- gray_z * train std",
            "gray_z": float(args.gray_z),
            "gray_zone_weight": float(args.gray_zone_weight),
            "threshold_mode": str(args.threshold_mode),
            "feature_preset": str(args.feature_preset),
            "include_exploratory_features": bool(args.include_exploratory_features),
            "freeze_adapter": bool(args.freeze_adapter),
            "freeze_dataset_bias": bool(args.freeze_dataset_bias),
            "experiment_name": str(args.experiment_name),
            "hidden_dim": int(args.hidden_dim),
            "adapter_dim": int(args.adapter_dim),
            "dropout": float(args.dropout),
            "global_loss_weight": float(args.global_loss_weight),
            "region_loss_weight": float(args.region_loss_weight),
            **pretrain_split_info,
        }
        pretrain_result = run_training_stage(
            stage_label=f"pretrain_{args.pretrain_source}",
            file_prefix=f"pretrain_{args.pretrain_source}",
            model=pretrain_model,
            train_loader=pretrain_train_loader,
            val_loader=pretrain_val_loader,
            train_ds=pretrain_train_ds,
            val_ds=pretrain_val_ds,
            device=device,
            pos_weight=pretrain_pos_weight,
            pos_weight_scalar=pretrain_pos_weight_scalar,
            epochs=args.pretrain_epochs,
            min_epochs=args.pretrain_min_epochs,
            patience_limit=args.pretrain_patience,
            lr=args.pretrain_lr,
            weight_decay=args.pretrain_weight_decay,
            global_loss_weight=args.global_loss_weight,
            region_loss_weight=args.region_loss_weight,
            grad_clip=args.grad_clip,
            save_dir=save_dir / f"pretrain_{args.pretrain_source}",
            config=pretrain_config,
        )

    finetune_train_loader, finetune_val_loader = make_loaders(
        finetune_train_ds,
        finetune_val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        device=device,
    )
    finetune_pos_weight, finetune_pos_weight_scalar = pos_weight_for_dataset(finetune_train_ds, device)
    print(
        f"[Loss] pos_weight={finetune_pos_weight_scalar:.4f} "
        f"global={args.global_loss_weight} region={args.region_loss_weight} gray_weight={args.gray_zone_weight}"
    )

    finetune_model = build_model(args, split_info, device)
    if pretrain_result is not None:
        pretrain_transfer = load_pretrained_into_finetune(
            pretrained_checkpoint_path=Path(pretrain_result["checkpoint"]),
            finetune_model=finetune_model,
            source_dataset_name=args.pretrain_source,
            finetune_dataset_order=list(split_info["dataset_order"]),
        )
        print(
            f"[Pretrain Transfer] checkpoint={pretrain_transfer['pretrained_checkpoint']} "
            f"shared_keys={pretrain_transfer['copied_shared_keys']} "
            f"adapter_copy={pretrain_transfer['adapter_copy']}"
        )
    apply_model_ablation_settings(finetune_model, args)

    finetune_config = {
        "model_family": "JointConstraintNet",
        "training_role": "cross_dataset_joint_constraint_binary_anxiety",
        "seed": int(seed),
        "device": str(device),
        "experiment_name": str(args.experiment_name),
        "features_root": str(Path(args.features_root)),
        "dataset_csvs": {name: str(path) for name, path in dataset_csvs.items()},
        "split_rule": "per_dataset_subject_level_split",
        "gray_zone_rule": "label by train median, weight gray band by train median +/- gray_z * train std",
        "gray_z": float(args.gray_z),
        "gray_zone_weight": float(args.gray_zone_weight),
        "threshold_mode": str(args.threshold_mode),
        "feature_preset": str(args.feature_preset),
        "include_exploratory_features": bool(args.include_exploratory_features),
        "freeze_adapter": bool(args.freeze_adapter),
        "freeze_dataset_bias": bool(args.freeze_dataset_bias),
        "hidden_dim": int(args.hidden_dim),
        "adapter_dim": int(args.adapter_dim),
        "dropout": float(args.dropout),
        "global_loss_weight": float(args.global_loss_weight),
        "region_loss_weight": float(args.region_loss_weight),
        "pretrain_source": str(args.pretrain_source),
        "pretrain_enabled": bool(args.pretrain_source != "none"),
        "pretrain_summary": None if pretrain_result is None else pretrain_result["summary"],
        "pretrain_transfer": pretrain_transfer,
        **split_info,
    }
    finetune_result = run_training_stage(
        stage_label="joint_constraint_finetune",
        file_prefix="joint_constraint",
        model=finetune_model,
        train_loader=finetune_train_loader,
        val_loader=finetune_val_loader,
        train_ds=finetune_train_ds,
        val_ds=finetune_val_ds,
        device=device,
        pos_weight=finetune_pos_weight,
        pos_weight_scalar=finetune_pos_weight_scalar,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience_limit=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        global_loss_weight=args.global_loss_weight,
        region_loss_weight=args.region_loss_weight,
        grad_clip=args.grad_clip,
        save_dir=save_dir,
        config=finetune_config,
    )

    summary = {
        "seed": int(seed),
        **{key: value for key, value in finetune_result["summary"].items() if key != "stage"},
        "used_pretrain": int(args.pretrain_source != "none"),
        "pretrain_source": str(args.pretrain_source),
    }
    if pretrain_result is not None:
        summary["pretrain_checkpoint"] = pretrain_result["checkpoint"]
        for key, value in pretrain_result["summary"].items():
            if key == "stage":
                continue
            summary[f"pretrain_{key}"] = value

    write_csv(save_dir / "summary_joint_constraint.csv", [summary])
    finetune_config["final_summary"] = summary
    write_json(save_dir / "config_joint_constraint.json", finetune_config)
    return {
        "summary": summary,
        "dataset_metrics": [
            {
                "seed": int(seed),
                **row,
            }
            for row in finetune_result["dataset_metrics"]
        ],
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[Device] {device}")
    print(f"[Output] {args.output_root.resolve()}")

    summaries = []
    dataset_metrics_all = []
    for seed in args.seeds:
        result = run_one_seed(args, int(seed), device)
        summaries.append(result["summary"])
        dataset_metrics_all.extend(result["dataset_metrics"])
    write_csv(args.output_root / "summary_all_seeds_joint_constraint.csv", summaries)
    write_csv(args.output_root / "summary_all_seeds_by_dataset_joint_constraint.csv", dataset_metrics_all)

    aggregate_rows, aggregate_payload = build_aggregate_payload(
        seeds=[int(seed) for seed in args.seeds],
        summary_rows=summaries,
        dataset_rows=dataset_metrics_all,
    )
    write_csv(args.output_root / "summary_aggregate_joint_constraint.csv", aggregate_rows)
    write_json(args.output_root / "summary_aggregate_joint_constraint.json", aggregate_payload)
    print_aggregate_report(aggregate_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
