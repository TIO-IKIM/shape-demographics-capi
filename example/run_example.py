"""Offline working example for shape-demographics (no downloads, ~1 minute).

Synthesises organ masks whose *shape* depends on a hidden "sex" (anisotropy) and
"age" (size), runs the REAL pipeline (marching cubes -> point cloud ->
correspondence-free descriptors -> cross-validated classifier/regressor), and
verifies that sex and age are recovered. This exercises the same code path used
for the papers, end to end, with no data download.

Run:  python example/run_example.py
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd

from shapedem.config import repo_root
from shapedem.shapes.extract import points_from_mask
from shapedem.shapes.descriptors import shape_descriptors
from shapedem.train.baselines import cv_classification, cv_regression


def ellipsoid_mask(radii_vox, shape=(64, 64, 64)):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = np.array(shape) / 2.0
    rz, ry, rx = radii_vox
    return ((zz - c[0]) / rz) ** 2 + ((yy - c[1]) / ry) ** 2 + ((xx - c[2]) / rx) ** 2 <= 1.0


def main(n=160, seed=0):
    rng = np.random.RandomState(seed)
    spacing = np.array([1.5, 1.0, 1.0])           # anisotropic voxels, like CT
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    rows = []
    for i in range(n):
        sex = i % 2                                # 1 = "female"
        age = rng.uniform(20, 80)
        row = {"subject": f"e{i:04d}", "y_sex": float(sex), "y_age": age}
        for organ, base in (("organA", 14.0), ("organB", 11.0)):
            elong = 1.0 + (0.35 if sex == 1 else 0.0) + rng.normal(0, 0.05)  # sex -> shape
            scale = 1.0 + (age - 50) / 200.0                                  # age -> size
            radii_mm = np.array([base * scale * (1 + rng.normal(0, 0.05)),
                                 base * scale,
                                 base * scale * elong])
            mask = ellipsoid_mask(radii_mm / spacing)
            sh = points_from_mask(mask, spacing, affine, n_points=1024,
                                  min_voxels=50, boundary_tol=10 ** 9)
            for k, v in shape_descriptors(sh).items():
                row[f"{organ}__{k}"] = v
            row[f"{organ}__present"] = 1.0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("subject")
    feat_cols = [c for c in df.columns if "__" in c and not c.endswith("__present")]
    X = df[feat_cols]
    sex = cv_classification(X, df["y_sex"].to_numpy(), kind="logreg")
    age = cv_regression(X, df["y_age"].to_numpy(), kind="ridge")

    print(f"Synthetic working example: {n} subjects, 2 organs, {len(feat_cols)} shape features")
    print(f"  sex : AUC {sex['auc']:.3f}  balanced-acc {sex['balanced_acc']:.3f}")
    print(f"  age : MAE {age['mae']:.2f} yr  R2 {age['r2']:.3f}")

    out = os.path.join(repo_root(), "_workspace", "example_output")
    os.makedirs(out, exist_ok=True)
    json.dump({"sex": sex, "age": age}, open(os.path.join(out, "example_results.json"), "w"), indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["sex AUC", "age $R^2$"], [sex["auc"], age["r2"]], color=["#3b7dd8", "#d8743b"])
    ax.set_ylim(0, 1.0); ax.set_title("Synthetic working example")
    fig.savefig(os.path.join(out, "example.png"), bbox_inches="tight", dpi=150); plt.close(fig)

    assert sex["auc"] > 0.7, "sex signal not recovered — the pipeline is broken"
    assert age["r2"] > 0.3, "age signal not recovered — the pipeline is broken"
    print(f"OK — pipeline works. Outputs in {out}/")


if __name__ == "__main__":
    main()
