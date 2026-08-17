"""Correspondence-free shape descriptors from a surface point cloud.

Called by features.py (to build the per-subject descriptor DataFrame) and
paperb.py (rater agreement analysis). Returns 7 size_ + 8 shape_ scalars
per organ instance.

SIZE features are scale-dependent; SHAPE features are computed after normalizing
the point cloud to unit centroid size, so they are scale-invariant. This lets
experiments switch between size-only / shape-only / full feature regimes via
features.select_columns().
"""
from __future__ import annotations
import numpy as np


def centroid_size(points: np.ndarray) -> float:
    c = points.mean(axis=0)
    return float(np.sqrt(((points - c) ** 2).sum()))


def _normalized(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    p = points - c
    cs = np.sqrt((p ** 2).sum())
    return p / cs if cs > 0 else p


def shape_descriptors(shape: dict) -> dict[str, float]:
    """Return a flat dict of named scalar features for one organ instance.

    Keys prefixed 'size_' are scale-dependent; 'shape_' are scale-invariant.
    """
    pts = np.asarray(shape["points"], dtype=np.float64)
    vol = float(shape.get("volume_mm3", np.nan))
    area = float(shape.get("surface_area_mm2", np.nan))

    cs = centroid_size(pts)
    # bounding box (size) and its scale-free aspect ratios (shape)
    ext = pts.max(0) - pts.min(0)
    ext_sorted = np.sort(ext)[::-1]

    # covariance eigenvalues on the SIZE-NORMALIZED cloud -> pure shape elongation
    q = _normalized(pts)
    cov = q.T @ q / len(q)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.clip(ev, 1e-12, None)

    feats: dict[str, float] = {}
    # --- SIZE features ---
    feats["size_centroid"] = cs
    feats["size_volume_mm3"] = vol
    feats["size_area_mm2"] = area
    feats["size_bbox_x"], feats["size_bbox_y"], feats["size_bbox_z"] = map(float, ext_sorted)
    feats["size_bbox_diag"] = float(np.sqrt((ext ** 2).sum()))

    # --- SHAPE features (scale-invariant) ---
    feats["shape_elong_21"] = float(ev[1] / ev[0])          # second/first PC variance
    feats["shape_elong_31"] = float(ev[2] / ev[0])          # third/first
    feats["shape_flatness"] = float(ev[2] / ev[1])
    feats["shape_aniso"] = float(1.0 - ev[2] / ev[0])
    feats["shape_aspect_yx"] = float(ext_sorted[1] / ext_sorted[0]) if ext_sorted[0] > 0 else np.nan
    feats["shape_aspect_zx"] = float(ext_sorted[2] / ext_sorted[0]) if ext_sorted[0] > 0 else np.nan
    # sphericity = pi^(1/3) (6V)^(2/3) / A  (1.0 = perfect sphere) — scale-invariant
    if vol > 0 and area > 0:
        feats["shape_sphericity"] = float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)
    else:
        feats["shape_sphericity"] = np.nan
    # surface-area-to-volume, made scale-free via centroid size
    feats["shape_sa_vol_norm"] = float(area / vol ** (2 / 3)) if vol > 0 else np.nan
    return feats


SIZE_PREFIX = "size_"
SHAPE_PREFIX = "shape_"
