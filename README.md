# Anxiety EEG Thesis Reproduction

面向弱标签、跨受试者 EEG 焦虑识别的论文复现仓库。

## Final Protocol

- 内部训练与验证：`EVA-MED`、OpenNeuro `ds003478`、OpenNeuro `ds007609`
- 外部兼容性验证：[Mendeley Data DOI 10.17632/sbyj5f6c3k.1](https://data.mendeley.com/datasets/sbyj5f6c3k/1)
- 划分规则：subject-level split；阈值、gray-zone 和标准化统计量仅由训练受试者估计
- 模型：轻量 MLP、dataset adapter、dataset bias 和辅助约束
- 基线：训练集内部交叉验证的 LogReg-L2、SVM 和树模型；显式可选 XGBoost

代码目录名 `original_local` 对应本地授权数据集 `EVA-MED`。

## Repository Layout

```text
src/anxiety_eeg/       Python 包源码
scripts/               常用命令入口
configs/               默认实验配置
data/                  数据下载与授权说明，不包含真实数据
tests/fixtures/        可提交的合成 smoke 数据
docs/                  复现流程、代码映射和结果边界
results/reference/     论文冻结表的轻量参考副本
outputs/               本地生成结果，已被 Git 忽略
```

## Quick Start

```powershell
python -m pip install -e .
python scripts/run_smoke.py --device cpu
```

smoke 使用合成特征表，执行 1 seed、1 epoch 的主模型训练和 LogReg-L2 基线训练。它只验证代码链路，不代表论文指标。

## Full Reproduction

将三套内部受试者级特征表放到：

```text
features/subject_features/original_local/subject_features.csv
features/subject_features/ds003478/subject_features.csv
features/subject_features/ds007609/subject_features.csv
```

运行主模型和 train-only CV 基线：

```powershell
python scripts/train_joint.py --config configs/default_joint.json
python scripts/train_baselines.py --features-root features/subject_features --models all
```

Mendeley 外部兼容性验证：

```powershell
python -m anxiety_eeg.external.extract_mendeley_subject_features --workbook data/mendeley_anxiety_control/EEG_data.xlsx
python -m anxiety_eeg.external.evaluate_external_mendeley --train-output-root outputs/joint_constraint --skip-scoring
```

完整步骤见 [docs/reproduction.md](docs/reproduction.md)，论文冻结结果见 [docs/results.md](docs/results.md)。

## Data Policy

本仓库不提交真实 EEG、标签表、Excel 工作簿、模型 checkpoint 或完整运行输出。公开数据入口：

- [OpenNeuro ds003478](https://openneuro.org/datasets/ds003478)
- [OpenNeuro ds007609](https://openneuro.org/datasets/ds007609)
- [Mendeley Data DOI 10.17632/sbyj5f6c3k.1](https://data.mendeley.com/datasets/sbyj5f6c3k/1)



## Interpretation Boundary

本研究不声称构建了高准确率临床诊断器。核心结论是：在严格 subject-level、弱标签、小样本和跨数据集域偏移条件下，EEG 全局频谱组织中存在弱但可检查的焦虑相关信号；模型提供有限稳定化，而非全面超越传统方法。

## License

当前仓库采用保守的 `All rights reserved` 声明。公开数据集仍遵循各自来源的许可与引用要求。
