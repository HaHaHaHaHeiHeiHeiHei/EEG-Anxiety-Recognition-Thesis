"""中文说明

用途：
    定义论文主模型 JointConstraintNet：共享 MLP 主干、dataset adapter、
    dataset bias，以及 global/frontal auxiliary heads。
输入：
    受试者级 EEG 全局频谱组织特征和数据集编号。
输出：
    焦虑高低分类 logit、全局辅助特征预测、额叶辅助特征预测和中间表示。
快速运行：
    通常由 `python scripts/train_joint.py` 调用；也可在 Python 中导入
    `from anxiety_eeg.models.joint_constraint import JointConstraintNet`。
论文对应：
    第 4 章“轻量 MLP 与约束结构”和第 5 章结构消融实验。
注意事项：
    该模型定位为小样本条件下的稳定化结构，不应解释为高容量深度网络。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointConstraintNet(nn.Module):
    """Shared trunk + dataset adapter + hierarchical auxiliary heads."""

    def __init__(
        self,
        input_dim: int,
        dataset_count: int,
        global_dim: int,
        region_dim: int,
        hidden_dim: int = 64,
        adapter_dim: int = 12,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.dataset_count = int(dataset_count)
        self.global_dim = int(global_dim)
        self.region_dim = int(region_dim)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.stem = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.dataset_embedding = nn.Embedding(self.dataset_count, adapter_dim)
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dataset_bias = nn.Embedding(self.dataset_count, 1)

        self.classifier = MLPHead(hidden_dim, hidden_dim, 1, dropout=dropout)
        self.global_head = MLPHead(hidden_dim, hidden_dim // 2, self.global_dim, dropout=dropout)
        self.region_head = MLPHead(hidden_dim, hidden_dim // 2, self.region_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, dataset_index: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared_trunk(self.stem(self.input_norm(x)))
        adapter = self.adapter(self.dataset_embedding(dataset_index))
        adapted = shared + adapter
        logit = self.classifier(adapted).squeeze(-1) + self.dataset_bias(dataset_index).squeeze(-1)
        global_pred = self.global_head(shared)
        region_pred = self.region_head(shared)
        return {
            "logit": logit,
            "global_pred": global_pred,
            "region_pred": region_pred,
            "shared_repr": shared,
            "adapted_repr": adapted,
        }
