# tests

本目录存放验证材料。当前重点是 `fixtures/` 中的合成 subject-level 特征表，用于 smoke run。

不在这里放真实 EEG 数据或论文实验输出。

## 输入

- `fixtures/subject_features` 中的合成 CSV。
- smoke 脚本和 `configs/smoke.json` 会读取这些 fixture。

## 输出

测试运行的输出写入 `outputs/smoke/` 或用户指定目录，不写回 `tests/`。

## 运行方式

```powershell
python scripts/run_smoke.py --device cpu
python scripts/train_joint.py --config configs/smoke.json
```

注意：这里的数据只验证代码可执行，不能用于论文指标。
