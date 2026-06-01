# features

本目录负责构建受试者级 EEG 频谱特征。

- `common_dataset_eval.py`：公共特征聚合与审计输出。
- `score_ds003478.py`：OpenNeuro `ds003478` 入口。
- `score_ds007609.py`：OpenNeuro/EEGDash `ds007609` 入口。
- `eegdash_remote_eval.py`：远程公开数据集缓存和提取工具。

真实数据和生成后的 `features/` 根目录不提交 Git。Mendeley Excel 兼容特征由 `anxiety_eeg.external.extract_mendeley_subject_features` 提取。
