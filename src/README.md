# src

Python 包源码目录。核心包为 `anxiety_eeg`，采用 `src` layout，便于 `pip install -e .` 后稳定导入。

## 使用方式

推荐在项目根目录执行：

```powershell
python -m pip install -e .
```

安装后可通过 `python -m anxiety_eeg.<module>` 调用包内模块。面向日常使用的入口优先放在 `scripts/`，本目录主要存放可复用实现。

## 输入与输出

- 输入通常来自 `features/subject_features`、`configs/` 或命令行参数。
- 输出统一写入 `outputs/` 或用户指定的 `--output-root`。
- 注意：本目录不存放真实 EEG 数据、处理结果或模型 checkpoint。
