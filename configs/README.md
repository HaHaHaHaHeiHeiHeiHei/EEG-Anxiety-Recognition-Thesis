# configs

本目录存放复现实验配置文件。训练脚本支持 `--config` 读取 JSON，键名与命令行参数一致。

- `smoke.json`：使用合成 fixture 的最小运行配置。
- `default_joint.json`：论文主模型的默认参数骨架，真实复现时需要把 `features_root` 指向完整特征表目录。

注意：配置不会自动下载数据，也不会把真实数据写入仓库。
