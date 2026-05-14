# 论文代码映射

| 论文内容 | 代码入口 | 说明 |
| --- | --- | --- |
| 受试者级特征表读取、无泄露划分 | `src/anxiety_eeg/data/joint_dataset.py` | 训练阈值、gray-zone、标准化均只由训练受试者估计 |
| JointConstraintNet 主模型 | `src/anxiety_eeg/models/joint_constraint.py` | shared MLP、dataset adapter、dataset bias、global/frontal auxiliary heads |
| 主模型训练 | `scripts/train_joint.py` | 对应 full_main / shared_subspace / 结构消融的核心训练 |
| 传统基线 | `scripts/train_baselines.py` | LogReg-L2、SVM、树模型等 |
| 消融实验 | `scripts/run_ablations.py` | no_adapter、no_global_aux、no_frontal_aux 等 |
| 共享子空间 logistic | `src/anxiety_eeg/analysis/run_shared_subspace_logistic.py` | 检查共有 theta/alpha/beta 特征子空间 |
| Mendeley 外部验证 | `src/anxiety_eeg/external/*mendeley*` | 部分特征重叠兼容性验证 |
| ds007216 外部审计 | `src/anxiety_eeg/external/*ds007216*` | domain-shift 和方向反转分析 |
| split-protocol control | `src/anxiety_eeg/analysis/compare_split_protocols.py` | 依赖本地授权原始数据，不属于默认 smoke |
