# tests/smoke

预留给后续自动化 smoke 测试脚本。当前 smoke 入口为：

```powershell
python scripts/run_smoke.py --device cpu
```

输入来自 `tests/fixtures/subject_features`，输出写入 `outputs/smoke/`。本目录目前不包含可执行测试文件，只作为后续扩展位置。

注意：smoke 只检查代码链路，不验证论文数值。
