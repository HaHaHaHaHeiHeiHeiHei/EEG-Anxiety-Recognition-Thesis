# training

本目录包含训练主入口。

- `train_joint.py`：训练 JointConstraintNet。
- `train_baselines.py`：训练传统机器学习基线。

推荐通过 `scripts/train_joint.py` 和 `scripts/train_baselines.py` 调用。

## 输入

- `features/subject_features` 下的三个内部数据集特征表。
- `configs/default_joint.json` 或命令行参数。

## 输出

- 主模型 summary：`summary_aggregate_joint_constraint.json`
- 传统基线 summary：`summary_aggregate_traditional_baselines.json`
- 默认写入 `outputs/` 或用户指定的 `--output-root`。

## 快速检查

```powershell
python scripts/train_joint.py --config configs/smoke.json
python scripts/train_baselines.py --features-root tests/fixtures/subject_features --output-root outputs/smoke/manual_baseline --seeds 42 --models logreg_l2 --n-jobs 1
```

注意：fixture 只用于验证训练代码可跑，不用于论文结果。
