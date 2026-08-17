"""Unit tests for the core shape pipeline (run with: pytest -q)."""
import numpy as np
import pandas as pd

from shapedem.shapes.extract import points_from_mask
from shapedem.shapes.descriptors import shape_descriptors, SIZE_PREFIX, SHAPE_PREFIX
from shapedem.shapes.ssm import umeyama
from shapedem.train.baselines import (cv_classification, cv_regression,
                                      oof_classification, bootstrap_ci,
                                      permutation_pvalue)
from shapedem.config import load_config


def _sphere(r=12, shape=(48, 48, 48)):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = np.array(shape) / 2.0
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r


def test_points_from_mask_shape_and_flag():
    sh = points_from_mask(_sphere(), np.ones(3), np.eye(4), n_points=512,
                          min_voxels=50, boundary_tol=10 ** 9)
    assert sh is not None
    assert sh["points"].shape == (512, 3)
    assert int(sh["truncated"]) == 0
    assert float(sh["volume_mm3"]) > 0


def test_min_voxels_returns_none():
    m = np.zeros((20, 20, 20), bool); m[0, 0, 0] = True
    assert points_from_mask(m, np.ones(3), np.eye(4), min_voxels=50) is None


def test_truncation_detected_at_boundary():
    big = _sphere(r=25, shape=(40, 40, 40))     # radius > half-extent -> clipped at the faces
    sh = points_from_mask(big, np.ones(3), np.eye(4), boundary_tol=20)
    assert int(sh["truncated"]) == 1


def test_sphere_descriptors_sane():
    d = shape_descriptors(points_from_mask(_sphere(), np.ones(3), np.eye(4),
                                           n_points=2048, boundary_tol=10 ** 9))
    assert 0.8 < d["shape_sphericity"] <= 1.05   # sphere ~ 1 (marching-cubes facets)
    assert d["shape_elong_21"] > 0.85            # near-isotropic
    assert any(k.startswith(SIZE_PREFIX) for k in d)
    assert any(k.startswith(SHAPE_PREFIX) for k in d)


def test_umeyama_recovers_similarity():
    rng = np.random.RandomState(0)
    src = rng.randn(100, 3)
    A = rng.randn(3, 3); R, _ = np.linalg.qr(A)
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    s, t = 2.0, np.array([1.0, -2.0, 3.0])
    dst = (s * (R @ src.T).T) + t
    s2, R2, t2 = umeyama(src, dst)
    assert abs(s2 - s) < 1e-6
    assert np.allclose(R2, R, atol=1e-6)
    assert np.allclose(t2, t, atol=1e-5)


def test_cv_classification_separable():
    rng = np.random.RandomState(0); n = 80
    X = pd.DataFrame({"f0": np.r_[rng.normal(0, 1, n // 2), rng.normal(4, 1, n // 2)]})
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    assert cv_classification(X, y, kind="logreg", n_splits=4)["auc"] > 0.9


def test_cv_regression_runs():
    rng = np.random.RandomState(0); n = 60
    x = rng.uniform(0, 1, n)
    X = pd.DataFrame({"f0": x + rng.normal(0, 0.01, n)})
    r = cv_regression(X, 10 * x, kind="ridge", n_splits=4)
    assert r["r2"] > 0.8


def test_oof_and_bootstrap_ci_separable():
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(0); n = 80
    X = pd.DataFrame({"f0": np.r_[rng.normal(0, 1, n // 2), rng.normal(4, 1, n // 2)]})
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    yt, prob = oof_classification(X, y, kind="logreg", n_splits=4)
    assert len(yt) == len(prob) == n
    mean, lo, hi = bootstrap_ci(yt, prob, roc_auc_score, n_boot=200)
    assert 0.9 < lo <= hi <= 1.0


def test_bootstrap_ci_empty_input_is_nan():
    from sklearn.metrics import roc_auc_score
    mean, lo, hi = bootstrap_ci(np.array([]), np.array([]), roc_auc_score, n_boot=10)
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_permutation_pvalue_detects_signal_and_null():
    rng = np.random.RandomState(0); n = 60
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    X_sig = pd.DataFrame({"f0": y * 4 + rng.normal(0, 1, n)})
    p_sig, obs_sig = permutation_pvalue(X_sig, y, kind="logreg", n_perm=30, n_splits=3)
    assert obs_sig > 0.9 and p_sig <= 2 / 31
    # under the null the observed AUC should be unremarkable; the p-value itself
    # is uniform under H0, so assert only its validity, not a threshold
    X_null = pd.DataFrame({"f0": rng.normal(0, 1, n)})
    p_null, obs_null = permutation_pvalue(X_null, y, kind="logreg", n_perm=30, n_splits=3)
    assert 0.3 < obs_null < 0.7
    assert 0 < p_null <= 1


def test_config_paths_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAPEDEM_ROOT", str(tmp_path))
    cfg = load_config()
    assert str(tmp_path) in cfg["paths"]["data_dir"]
    assert "${" not in cfg["paths"]["data_dir"]      # tokens fully substituted
