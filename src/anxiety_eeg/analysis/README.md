# analysis

- `run_joint_ablation_suite.py`：结构消融与 gray-zone 敏感性。
- `run_shared_subspace_logistic.py`：共享 theta/alpha/beta 子空间 logistic 参考。

```powershell
python scripts/run_ablations.py --features-root features/subject_features --skip-external
python -m anxiety_eeg.analysis.run_shared_subspace_logistic --help
```
