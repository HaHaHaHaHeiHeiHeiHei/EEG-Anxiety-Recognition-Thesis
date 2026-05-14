# original_local / EVA-MED 本地队列

用途：论文内部本地队列，标签为 `tra_anx`，用于主模型内部训练与验证。

获取方式：该队列属于课题组/本地授权数据，当前仓库不提供公开下载链接，也不放置真实数据文件。需要复现时，请向数据负责人获取授权副本。

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
