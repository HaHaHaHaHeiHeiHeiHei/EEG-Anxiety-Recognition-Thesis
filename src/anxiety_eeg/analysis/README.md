# analysis

本目录包含论文分析脚本。

- `run_joint_ablation_suite.py`：结构消融。
- `run_shared_subspace_logistic.py`：共享子空间 logistic 参考。
- `analyze_direction_consistency.py`：跨数据集特征方向一致性。
- `compare_split_protocols.py`：segment-random 与 subject-independent 对照，依赖本地授权原始数据。

## 输入

- 主模型或基线输出目录。
- `features/subject_features` 或外部数据特征表。
- 少数旧对照脚本可能需要本地授权原始数据。

## 输出

分析结果默认写入 `outputs/` 下的对应实验目录，包括 CSV、JSON 或控制台汇总。

## 运行方式

```powershell
python scripts/run_ablations.py --features-root features/subject_features --output-root outputs/joint_ablation_suite --skip-external
python -m anxiety_eeg.analysis.run_shared_subspace_logistic --help
python -m anxiety_eeg.analysis.analyze_direction_consistency --help
```

注意：分析脚本用于解释稳定性、方向一致性和泄露风险，不应替代主训练结果。
