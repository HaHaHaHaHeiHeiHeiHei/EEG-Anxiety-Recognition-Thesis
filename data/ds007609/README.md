# ds007609

用途：论文内部联合建模数据集之一，使用 trait anxiety 相关标签和 EEG 频谱特征。

下载链接：[OpenNeuro ds007609](https://openneuro.org/datasets/ds007609)

建议流程：

```powershell
python -m anxiety_eeg.features.score_ds007609 --output-dir features/subject_features/ds007609
```

输出位置应为 `features/subject_features/ds007609/subject_features.csv`，供主模型和传统基线读取。

本目录只放说明，不放真实数据。首次提取可能需要下载并缓存公开 EEG 文件。
