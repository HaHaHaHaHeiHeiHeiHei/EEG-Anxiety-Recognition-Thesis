# anxiety_eeg

- `data/`：受试者级特征读取、标签阈值、gray-zone 和 subject-level split。
- `features/`：公开 EEG 与 Mendeley 到 subject-level 特征表的工具。
- `models/`：JointConstraintNet。
- `training/`：主模型和传统基线训练。
- `analysis/`：正式消融与共享子空间参考。
- `external/`：Mendeley 外部兼容性验证。

优先使用根目录 `scripts/` 入口。包内模块适合特征提取、分析和外部验证。
