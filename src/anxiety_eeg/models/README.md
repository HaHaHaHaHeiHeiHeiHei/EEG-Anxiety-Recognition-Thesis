# models

本目录存放模型结构。

- `joint_constraint.py`：JointConstraintNet，包含共享 MLP、dataset adapter、dataset bias、global auxiliary head 和 frontal/region auxiliary head。

该模型用于论文第 4 章和第 5 章主模型/消融实验。

## 输入与输出

- 输入：由 `data/joint_dataset.py` 生成的受试者级 EEG 特征张量、数据集编号和标签。
- 输出：主任务 logits、全局辅助约束输出、区域辅助约束输出。

## 使用方式

一般不直接运行本目录文件，而是通过：

```powershell
python scripts/train_joint.py --config configs/default_joint.json
```

注意：模型文件只定义结构，不保存 checkpoint；训练产物应写入 `outputs/`。
