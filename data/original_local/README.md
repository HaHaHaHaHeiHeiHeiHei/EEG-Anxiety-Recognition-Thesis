# original_local / EVA-MED

用途：论文内部训练与验证数据集，标签为 `tra_anx`。

下载入口：[EVA-MED: An Enhanced Valence-Arousal Multimodal Emotion Dataset for Emotion Recognition](https://www.scidb.cn/detail?dataSetId=e15a1364db5f425889d6d631055c8420)

说明：下载和使用时请遵循 SciDB 数据页面列出的许可与引用要求。仓库中的 `original_local` 名称是为兼容既有特征表、checkpoint 元数据和训练脚本保留的稳定代码标识。

期望结构示例：

```text
data/original_local/
  raw/
  labels/PI_emo_tra_anx.csv
```

主模型不直接读取原始数据，而读取提取后的：

```text
features/subject_features/original_local/subject_features.csv
```

输出位置应为 `features/subject_features/original_local/subject_features.csv`，供主模型和传统基线读取。

说明：请不要把原始 FIF、CSV 标签表或个人敏感信息提交到 Git。
