# Mendeley Anxiety/Control

用途：论文 Mendeley 外部兼容性验证，标签为 anxiety/control 二分类。

下载链接：[Mendeley Data DOI 10.17632/sbyj5f6c3k.1](https://data.mendeley.com/datasets/sbyj5f6c3k/1)

建议将官方下载工作簿放为：

```text
data/mendeley_anxiety_control/EEG_data.xlsx
```

然后运行：

```powershell
python -m anxiety_eeg.external.extract_mendeley_subject_features --workbook data/mendeley_anxiety_control/EEG_data.xlsx
```

输出默认写入 `features/external/mendeley/subject_features.csv`，供 `evaluate_external_mendeley` 读取；也可用 `--output-dir` 指定其他本地目录。

说明：本目录只放下载说明，不提交工作簿。Mendeley 缺少训练特征中的 delta、gamma 和 `beta/(delta+theta)`，因此只作为 partial-overlap compatibility test。
