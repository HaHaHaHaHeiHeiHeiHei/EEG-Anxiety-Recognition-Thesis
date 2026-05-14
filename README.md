# 面向焦虑情绪诱发与识别的 EEG 复现实验代码

本仓库用于实现“弱标签、跨受试者 EEG 焦虑识别”的主线实验。

## 研究主线

论文开题阶段曾实现 EEG、ECG、PI 多模态同步分析，但实验增益不稳定，因此最终主体收敛为 EEG 单模态：

- 受试者级 EEG 全局频谱组织特征
- 无泄露 subject-level train/validation split
- 轻量 MLP + dataset adapter + auxiliary constraints
- 固定超参数传统机器学习基线
- 表征/结构消融
- Mendeley 外部兼容性验证
- ds007216 domain-shift / direction mismatch 审计

## 项目结构

```text
src/anxiety_eeg/       Python 包源码
scripts/               常用命令入口
configs/               JSON 实验配置
data/                  只放数据集名称文件夹和下载/获取说明
tests/fixtures/        合成 smoke 数据
docs/                  复现说明和论文代码映射
outputs/               默认输出目录，不提交 Git
```

## 安装

```powershell
python -m pip install -e .
python -m pip install -r requirements.txt
```

## 完整执行流程

第一次检查或复现实验时，建议先阅读 [执行流程介绍.md](执行流程介绍.md)。该文档按顺序说明环境安装、smoke 验证、真实数据准备、特征提取、主模型训练、传统基线、消融实验、外部验证和输出检查。

## 快速验证

```powershell
python scripts/run_smoke.py --device cpu
```

该命令使用合成特征表，完成 1 seed、1 epoch 的主模型训练和 LogReg-L2 基线训练。它只验证代码可跑，不代表论文指标。

## 真实复现

1. 按 `data/<dataset>/README.md` 准备数据。
2. 将受试者级特征表放到：

```text
features/subject_features/original_local/subject_features.csv
features/subject_features/ds003478/subject_features.csv
features/subject_features/ds007609/subject_features.csv
```

3. 训练主模型：

```powershell
python scripts/train_joint.py --config configs/default_joint.json
```

4. 训练传统基线：

```powershell
python scripts/train_baselines.py --features-root features/subject_features --models all
```

说明：`--models all` 只运行默认传统基线，包括 LogReg-L2、Linear SVM、RBF SVM、Random Forest、Extra Trees 和 Gradient Boosting。XGBoost 是可选扩展模型，默认不运行；如需运行，请先安装 `xgboost`，再显式执行 `--models xgboost`。

完整执行顺序见 `执行流程介绍.md`，精简复现命令见 `docs/reproduction.md`。

## 数据说明

本仓库不放真实数据。公开数据入口：

- [OpenNeuro ds003478](https://openneuro.org/datasets/ds003478)
- [OpenNeuro ds007609](https://openneuro.org/datasets/ds007609)
- [OpenNeuro ds007216](https://openneuro.org/datasets/ds007216)
- [Mendeley Data DOI 10.17632/sbyj5f6c3k.1](https://data.mendeley.com/datasets/sbyj5f6c3k/1)

`original_local / EVA-MED` 为课题组/本地授权数据，不伪造公开下载链接。

## 解释边界

本研究不声称构建了高准确率临床诊断器。核心结论是：在严格 subject-level、弱标签、小样本和跨数据集域偏移条件下，EEG 全局频谱组织中存在弱但可检查的焦虑相关信号；模型提供的是有限稳定化，而非全面超越传统方法。

## 与论文表述的一致性说明

- 主模型默认配置使用 25 个随机种子，与论文第 5 章的多 seed 报告一致。
- 传统机器学习基线使用固定超参数，并复用与主模型一致的 subject-level split、训练集阈值和 gray-zone 标签规则。
- 若论文中写到传统基线，建议表述为“固定超参数传统基线”或“与主模型协议一致的传统基线”，不要写成 “train-only CV”，除非后续另行加入内层交叉验证。
