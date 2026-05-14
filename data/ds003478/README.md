# ds003478

用途：论文内部联合建模数据集之一，使用 STAI 相关标签和 EEG 频谱特征。

下载链接：[OpenNeuro ds003478](https://openneuro.org/datasets/ds003478)

建议流程：

```powershell
python -m anxiety_eeg.features.score_ds003478 --ds-root data/ds003478 --output-dir features/subject_features/ds003478
```

输出位置应为 `features/subject_features/ds003478/subject_features.csv`，供主模型和传统基线读取。

本目录只放说明，不放真实数据。下载和使用时请遵守 OpenNeuro 数据许可与引用要求。
