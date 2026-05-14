# ds007609 fixture

合成的 ds007609 风格特征表，仅用于 smoke 测试。

文件：

- `subject_features.csv`：8 个虚拟受试者，列名与主模型默认输入一致。

输入：无，文件已随仓库提供。输出：运行 smoke 后写入 `outputs/smoke/`，本目录不被修改。

注意：该文件只用于可跑性检查，不能用于论文指标。

运行方式：

```powershell
python scripts/run_smoke.py --device cpu
```
