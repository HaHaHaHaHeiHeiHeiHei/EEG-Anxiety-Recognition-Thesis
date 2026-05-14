# data

本目录负责读取 `subject_features.csv`、构造无泄露 subject-level 训练/验证划分，并生成 PyTorch Dataset。

核心文件：

- `joint_dataset.py`：主模型和传统基线共享的数据构建逻辑。

默认真实特征根目录为 `features/subject_features`，smoke 测试使用 `tests/fixtures/subject_features`。

## 输入

- `features/subject_features/<dataset>/subject_features.csv`
- 必要列包括 `dataset`、`subject`、`score_name`、`anxiety` 和模型需要的特征列。

## 输出

- 训练/验证划分后的内存数据结构。
- PyTorch `Dataset` 和传统基线可使用的特征矩阵。

本目录只负责数据读取和适配，不保存真实数据、不写入输出文件。
