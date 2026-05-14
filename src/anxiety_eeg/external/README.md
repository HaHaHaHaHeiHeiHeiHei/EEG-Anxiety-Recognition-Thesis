# external

本目录包含外部验证脚本。

- `extract_mendeley_subject_features.py`：从 Mendeley Excel 提取兼容特征。
- `evaluate_external_mendeley.py`：Mendeley partial compatibility 外部评估。
- `evaluate_external_ds007216.py`：ds007216 domain-shift 外部审计。
- `analyze_reverse_discrimination_ds007216.py`：ds007216 反向判别分析。

## 输入

- 已训练主模型输出目录，例如 `outputs/joint_constraint`。
- Mendeley 或 ds007216 的外部特征文件。

## 输出

外部验证指标、方向审计结果和兼容性说明，默认写入 `outputs/` 或脚本参数指定目录。

## 运行方式

```powershell
python -m anxiety_eeg.external.evaluate_external_mendeley --help
python -m anxiety_eeg.external.evaluate_external_ds007216 --help
python -m anxiety_eeg.external.analyze_reverse_discrimination_ds007216 --help
```

注意：外部验证用于 partial compatibility 和 domain-shift 审计，不作为主模型最终性能的唯一证据。
