# scripts

- `run_smoke.py`：合成 fixture 最小运行检查。
- `prepare_features.py`：打印真实特征准备路径。
- `train_joint.py`：主模型训练入口。
- `train_baselines.py`：train-only CV 传统基线入口。
- `run_ablations.py`：结构消融入口。

所有本地输出默认写入 Git 忽略的 `outputs/`。
