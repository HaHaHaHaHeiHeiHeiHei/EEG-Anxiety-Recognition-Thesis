# ds007216

用途：论文外部 domain-shift 审计数据集，用于方向一致性和反向判别分析。

下载链接：[OpenNeuro ds007216](https://openneuro.org/datasets/ds007216)

建议流程：

```powershell
python -m anxiety_eeg.features.score_ds007216 --output-dir features/external/ds007216
```

输出位置应为 `features/external/ds007216/subject_features.csv` 或同目录下外部评估脚本约定的特征文件。

注意：该数据集不是主训练集，论文中用于外部敏感性分析，不应被解释为主模型最终泛化性能。
