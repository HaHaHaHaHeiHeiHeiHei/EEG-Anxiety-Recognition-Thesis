# data

本目录只存放数据集名称文件夹和下载/获取说明，不存放真实 EEG 数据、处理后特征表、Excel 工作簿或模型文件。

真实复现时，请按各子目录 `README.md` 下载或准备原始数据，然后将提取后的受试者级特征表放到：

```text
features/subject_features/<dataset>/subject_features.csv
```

smoke 测试使用 `tests/fixtures/subject_features` 中的合成数据。

## 输入

- 内部公开数据集：按 `original_local/`、`ds003478/`、`ds007609/` 中的链接自行下载。
- `original_local/` 是 `EVA-MED` 的稳定代码目录名。
- Mendeley：按 `mendeley_anxiety_control/` 的 DOI 链接下载工作簿。

## 输出

本目录不产生也不保存输出。特征提取结果应写入 `features/`，训练和分析结果应写入 `outputs/`。

## 注意事项

请不要把原始 EEG 文件、真实标签表、Excel 工作簿、处理后特征表或个人敏感信息提交到 Git。
