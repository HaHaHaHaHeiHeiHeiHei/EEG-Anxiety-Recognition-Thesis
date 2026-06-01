# Contributing

## Local Check

提交前运行：

```powershell
python -m compileall -q src scripts
python scripts/run_smoke.py --device cpu
git diff --check
```

## Data Safety

不要提交真实 EEG、标签表、Excel 工作簿、模型 checkpoint 或 `outputs/` 目录。公开问题报告中也不要粘贴受试者级敏感信息。

## Scope

本仓库只维护终稿复现主线。新增实验应说明其数据协议、是否属于正式结果，以及是否会改变训练集内部统计量。
