# external

- `extract_mendeley_subject_features.py`：从官方下载 Excel 提取 theta/alpha/beta 兼容特征。
- `evaluate_external_mendeley.py`：加载已训练 checkpoint 进行 Mendeley partial-overlap compatibility test。

```powershell
python -m anxiety_eeg.external.extract_mendeley_subject_features --help
python -m anxiety_eeg.external.evaluate_external_mendeley --help
```

缺失训练特征按 pooled training mean 中性填补。外部结果不应解释为完整特征空间上的强泛化结论。
