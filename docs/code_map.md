# 论文代码映射

| 论文内容 | 代码入口 | 说明 |
| --- | --- | --- |
| 受试者级特征读取与无泄露划分 | `src/anxiety_eeg/data/joint_dataset.py` | split、训练阈值、gray-zone、标准化 |
| JointConstraintNet | `src/anxiety_eeg/models/joint_constraint.py` | shared MLP、adapter、dataset bias、辅助约束 |
| 主模型训练 | `scripts/train_joint.py` | 25 seeds 正式入口 |
| 传统基线 | `scripts/train_baselines.py` | train-only CV 与可选 XGBoost |
| 结构消融 | `scripts/run_ablations.py` | adapter、辅助约束、gray-zone 敏感性 |
| 共享子空间 logistic | `src/anxiety_eeg/analysis/run_shared_subspace_logistic.py` | theta/alpha/beta 共有子空间参考 |
| Mendeley 特征提取 | `src/anxiety_eeg/external/extract_mendeley_subject_features.py` | Excel 到兼容 subject-level 特征 |
| Mendeley 外部评估 | `src/anxiety_eeg/external/evaluate_external_mendeley.py` | partial-overlap compatibility test |
