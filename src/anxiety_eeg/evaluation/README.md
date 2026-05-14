# evaluation

评估工具预留目录。当前项目的主要评估逻辑分布在：

- `training/`：训练过程中的验证指标和聚合 summary。
- `external/`：Mendeley、ds007216 外部评估。
- `analysis/`：方向一致性、消融和 split-protocol 分析。

## 输入与输出

当前目录暂不提供独立入口。评估输入来自训练或外部验证脚本，输出通常为 `outputs/` 下的 summary JSON、CSV 指标表或控制台报告。

## 使用方式

请优先运行：

```powershell
python scripts/run_smoke.py --device cpu
python scripts/train_joint.py --config configs/default_joint.json
python scripts/train_baselines.py --features-root features/subject_features --models all
```

注意：如果后续新增共享评估函数，应在本目录 README 同步记录调用方式和输出格式。
