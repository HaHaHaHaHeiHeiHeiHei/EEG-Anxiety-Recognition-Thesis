# 最终数据协议

## 数据划分

| 用途 | 数据集 |
| --- | --- |
| 内部训练与验证 | `EVA-MED`、OpenNeuro `ds003478`、OpenNeuro `ds007609` |
| 外部兼容性验证 | Mendeley Data DOI `10.17632/sbyj5f6c3k.1` |

代码中的 `original_local` 标识对应本地授权数据集 `EVA-MED`。

## 无泄露规则

- 按受试者划分训练集和验证集。
- 每个内部数据集的标签阈值由训练受试者中位数估计。
- gray-zone 为训练中位数 `± gray_z * train_std`。
- 标准化统计量只由训练受试者估计。
- 传统基线的超参数选择只在 outer-train 内部交叉验证中完成。

## 公开仓库边界

本仓库不包含真实 EEG、授权标签、处理后真实特征表或 checkpoint。Mendeley 只共享 theta、alpha、beta 可用信息；缺失训练特征在外部加载时使用 pooled training mean 中性填补。
