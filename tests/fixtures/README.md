# tests/fixtures

这里存放极小合成数据，只用于验证代码链路能跑通。它不是论文真实数据，也不能用于报告论文指标。

`subject_features/` 的结构模拟真实特征目录：

```text
subject_features/
  original_local/subject_features.csv
  ds003478/subject_features.csv
  ds007609/subject_features.csv
```

每个 CSV 只有 8 个合成受试者，包含主模型默认需要的全局频谱和额叶特征列。

## 输入与输出

本目录本身不需要外部输入，也不产生输出。运行 smoke 后，结果应写入 `outputs/smoke/`。

## 运行方式

```powershell
python scripts/run_smoke.py --device cpu
```
