# scripts

面向用户的薄入口脚本。

- `run_smoke.py`：使用合成 fixture 验证主模型和 LogReg-L2 可跑。
- `train_joint.py`：训练论文主模型。
- `train_baselines.py`：训练传统基线。
- `run_ablations.py`：运行结构消融。
- `prepare_features.py`：打印公开数据集特征提取命令。

所有脚本均可在未安装 editable 包时直接通过 `python scripts/<name>.py` 调用。

## 输入

- 默认真实特征根目录为 `features/subject_features`。
- smoke 验证使用 `tests/fixtures/subject_features`。
- 配置文件位于 `configs/`，可通过 `--config` 传入。

## 输出

- 训练、基线和消融结果默认写入 `outputs/`。
- 本目录不存放输出文件，也不写入真实数据目录。

## 常用命令

```powershell
python scripts/run_smoke.py --device cpu
python scripts/train_joint.py --config configs/default_joint.json
python scripts/train_baselines.py --features-root features/subject_features --models all
python scripts/run_ablations.py --features-root features/subject_features --skip-external
python scripts/prepare_features.py
```

注意：`run_smoke.py` 只证明代码链路可跑，不代表论文实验指标。
