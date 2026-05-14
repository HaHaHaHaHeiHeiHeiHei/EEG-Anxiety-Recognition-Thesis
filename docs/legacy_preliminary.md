# 前期多模态与旧路线说明

论文开题阶段包含 EEG、ECG、PI 多模态同步分析，工程上完成了文件检查、EEG 基准对齐、PI 重采样、滑窗切分、特征提取和多输入组合训练。

最终论文主体收敛到 EEG 单模态，原因是前期多模态实验只显示有限且不稳定的 RMSE 改善，R² 多数为负，ECG/PI 在当前样本量和标签质量下没有形成足够稳定的主结论证据。

因此，本仓库默认复现以下主线：

- EEG subject-level global spectral organization
- JointConstraintNet
- traditional baselines
- representation / structure ablations
- Mendeley compatibility validation
- ds007216 domain-shift audit

旧的 DEAP、PI-only、late-fusion、多模态时序模型不作为默认入口。若需要追溯，可回到原始研发目录 `dataset_code_11` 查找历史实验，但不要把原始数据、checkpoint 或大输出提交到本仓库。
