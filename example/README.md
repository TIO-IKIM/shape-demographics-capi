# Working example

`run_example.py` is a self-contained, **offline** sanity check (no downloads, ~1
minute). It synthesises organ masks whose *shape* depends on a hidden sex
(anisotropy) and age (size), then runs the **real** pipeline used for the papers
— marching cubes → surface point cloud → correspondence-free descriptors →
cross-validated classifier/regressor — and asserts that sex and age are recovered.

```bash
source .venv/bin/activate         # created by ../setup.sh
python example/run_example.py
```

Expected output (deterministic, seed=0):
```
Synthetic working example: 160 subjects, 2 organs, 30 shape features
  sex : AUC 1.000  balanced-acc 1.000
  age : MAE 3.04 yr  R2 0.944
OK — pipeline works. Outputs in _workspace/example_output/
```
It writes `_workspace/example_output/example_results.json` and `example.png`.

To run the pipeline on **real data** instead, see the top-level `README.md`
(`bash scripts/download_capi.sh`, then `python -m shapedem.cli paperb --step all`).
