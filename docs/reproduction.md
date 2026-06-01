# 复现流程

## 1. 安装

```powershell
python -m pip install -e .
```

若使用 conda：

```powershell
conda env create -f environment.yml
conda activate anxiety-eeg
python -m pip install -e .
```

## 2. smoke 检查

```powershell
python scripts/run_smoke.py --device cpu
```

该命令读取 `tests/fixtures/subject_features` 合成数据，预期生成：

- `outputs/smoke/joint/summary_aggregate_joint_constraint.json`
- `outputs/smoke/baselines/summary_aggregate_traditional_baselines.json`

## 3. 准备真实特征

`EVA-MED` 下载入口：

- [SciDB EVA-MED 数据页面](https://www.scidb.cn/detail?dataSetId=e15a1364db5f425889d6d631055c8420)

```text
features/subject_features/original_local/subject_features.csv  # EVA-MED
features/subject_features/ds003478/subject_features.csv
features/subject_features/ds007609/subject_features.csv
```

每个 CSV 至少包含 `dataset, score_name, subject, anxiety` 和默认频谱特征列。

可用入口：

```powershell
python scripts/prepare_features.py
python -m anxiety_eeg.features.score_ds003478 --help
python -m anxiety_eeg.features.score_ds007609 --help
```

`ds007609` 远程入口可选依赖 `eegdash`。`ds003478` 原始 EEG 重建仍需要本地数据 helper；公开仓库保证训练接口和 Mendeley 提取器可检查，不提交授权数据 helper。

## 4. 训练主模型

```powershell
python scripts/train_joint.py --config configs/default_joint.json
```

单 seed 快速检查：

```powershell
python scripts/train_joint.py --features-root features/subject_features --output-root outputs/check_joint --seeds 42 --epochs 1 --min-epochs 1 --patience 1 --device cpu
```

## 5. 训练传统基线

```powershell
python scripts/train_baselines.py --features-root features/subject_features --output-root outputs/traditional_baselines --models all
```

`--models all` 运行 LogReg-L2、Linear SVM、RBF SVM、Random Forest、Extra Trees 和 Gradient Boosting。它们默认使用 outer-train 内部交叉验证。XGBoost 为显式固定参数扩展：

```powershell
python -m pip install xgboost
python scripts/train_baselines.py --features-root features/subject_features --output-root outputs/traditional_baselines_xgboost --models xgboost
```

## 6. 消融

```powershell
python scripts/run_ablations.py --features-root features/subject_features --output-root outputs/joint_ablation_suite --skip-external
```

## 7. Mendeley 外部兼容性验证

```powershell
python -m anxiety_eeg.external.extract_mendeley_subject_features --workbook data/mendeley_anxiety_control/EEG_data.xlsx
python -m anxiety_eeg.external.evaluate_external_mendeley --train-output-root outputs/joint_constraint --skip-scoring
```

Mendeley 缺少部分训练特征，因此该结果应解释为 partial-overlap compatibility test。

## 8. 提交前检查

```powershell
python -m compileall -q src scripts
python scripts/run_smoke.py --device cpu
git diff --check
```
