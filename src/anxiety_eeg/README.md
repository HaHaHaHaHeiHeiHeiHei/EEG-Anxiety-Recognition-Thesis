# anxiety_eeg

论文复现实验主包。

- `data/`：受试者级特征表读取、标签阈值、gray-zone、subject-level split。
- `features/`：公开 EEG 原始数据到 subject-level 频谱特征表的提取工具。
- `models/`：JointConstraintNet 模型结构。
- `training/`：主模型和传统基线训练。
- `analysis/`：消融、共享子空间、方向一致性、split-protocol control。
- `external/`：Mendeley 和 ds007216 外部验证。
- `legacy/`：本地授权旧路线占位，不随公开仓库分发真实数据。

## 运行入口

优先使用根目录 `scripts/`：

```powershell
python scripts/run_smoke.py --device cpu
python scripts/train_joint.py --config configs/default_joint.json
python scripts/train_baselines.py --features-root features/subject_features --models all
```

包内模块也可在安装后用 `python -m anxiety_eeg...` 调用，常用于特征提取、分析和外部验证。

## 输入与输出

- 输入：`subject_features.csv`、JSON 配置和公开/授权数据说明中的本地文件。
- 输出：训练 summary、分析 CSV/JSON、外部验证结果，默认写入 `outputs/`。
- 注意：本包不内置真实数据；合成 fixture 只用于 smoke 检查。
