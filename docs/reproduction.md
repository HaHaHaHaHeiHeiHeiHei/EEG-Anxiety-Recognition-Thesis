# 复现流程

## 1. 安装环境

```powershell
python -m pip install -e .
python -m pip install -r requirements.txt
```

## 2. 验证代码可跑

```powershell
python scripts/run_smoke.py --device cpu
```

该命令使用 `tests/fixtures/subject_features` 的合成数据，预期生成：

- `outputs/smoke/joint/summary_aggregate_joint_constraint.json`
- `outputs/smoke/baselines/summary_aggregate_traditional_baselines.json`

## 3. 准备真实特征

公开仓库不放真实数据。真实复现时，将三套内部训练特征放成：

```text
features/subject_features/original_local/subject_features.csv
features/subject_features/ds003478/subject_features.csv
features/subject_features/ds007609/subject_features.csv
```

每个 CSV 至少包含 `dataset, score_name, subject, anxiety` 和默认频谱特征列。

## 4. 训练主模型

```powershell
python scripts/train_joint.py --config configs/default_joint.json
```

默认配置使用 25 个随机种子。单 seed 快速检查：

```powershell
python scripts/train_joint.py --features-root features/subject_features --output-root outputs/check_joint --seeds 42 --epochs 1 --min-epochs 1 --patience 1 --device cpu
```

## 5. 训练传统基线

```powershell
python scripts/train_baselines.py --features-root features/subject_features --output-root outputs/traditional_baselines --models all
```

`--models all` 只运行默认无额外依赖的传统基线：

- `logreg_l2`
- `linear_svm`
- `rbf_svm`
- `random_forest`
- `extra_trees`
- `gradient_boosting`

这些基线使用固定超参数，并复用与主模型一致的 subject-level split、训练集阈值、gray-zone 标签和输入特征。当前版本没有实现内层交叉验证，因此论文中不建议写成 “train-only CV”。

如需额外运行 XGBoost：

```powershell
python -m pip install xgboost
python scripts/train_baselines.py --features-root features/subject_features --output-root outputs/traditional_baselines_xgboost --models xgboost
```

## 6. 运行消融

```powershell
python scripts/run_ablations.py --features-root features/subject_features --output-root outputs/joint_ablation_suite --skip-external
```

## 7. 外部验证

Mendeley：

```powershell
python -m anxiety_eeg.external.extract_mendeley_subject_features --workbook data/mendeley_anxiety_control/EEG_data.xlsx
python -m anxiety_eeg.external.evaluate_external_mendeley --train-output-root outputs/joint_constraint --skip-scoring
```

ds007216：

```powershell
python -m anxiety_eeg.features.score_ds007216 --output-dir features/external/ds007216
python -m anxiety_eeg.external.evaluate_external_ds007216 --train-output-root outputs/joint_constraint --skip-scoring
python -m anxiety_eeg.external.analyze_reverse_discrimination_ds007216
```

## 8. 输出检查

建议检查：

```powershell
python -m compileall src scripts tests
python scripts/train_joint.py --help
python scripts/train_baselines.py --help
python scripts/run_ablations.py --help
```

合成 fixture 只验证流程可跑，不用于复现论文数值。论文数值需要真实授权特征表、默认 25 seeds 和完整训练配置。
