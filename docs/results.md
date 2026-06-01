# 结果参考

## 论文冻结结果

论文表格的轻量 CSV 副本位于：

- `results/reference/table_5_1_internal_comparison.csv`
- `results/reference/table_5_8_mendeley_compatibility.csv`

主结果：

| 模型 | Joint score | ROC-AUC | PR-AUC | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| Proposed Full | `0.635 ± 0.040` | `0.589 ± 0.052` | `0.621 ± 0.054` | `0.593 ± 0.038` |
| LogReg-L2 + train-only CV | `0.609 ± 0.033` | `0.577 ± 0.049` | `0.637 ± 0.051` | `0.568 ± 0.045` |

Mendeley partial-overlap compatibility：

| 模型 | ROC-AUC | PR-AUC | Balanced Accuracy |
| --- | --- | --- | --- |
| Proposed Full Model (mean adapter) | `0.615 ± 0.027` | `0.659 ± 0.037` | `0.571 ± 0.049` |
| Extra Trees | `0.618 ± 0.029` | `0.668 ± 0.029` | `0.535 ± 0.048` |

## 复现说明

神经网络在不同 PyTorch、CPU 数值环境和早停 epoch 下可能出现小幅 seed-level 浮动。论文引用应使用冻结表；本地复跑用于验证代码路径和结果量级。合成 smoke fixture 不得作为论文实验结果。
