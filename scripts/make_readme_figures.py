#!/usr/bin/env python
"""Generate the README illustration figures (not used by the paper).

Outputs (docs/assets/):
  pipeline_umd.png    4-panel pipeline illustration on one UMD subject
                      (T2 slice -> segmentation overlay -> marching-cubes
                      mesh -> 2,048-point surface cloud). UMD is CC BY 4.0,
                      so showing one subject's slice is license-compatible;
                      UT-EndoMRI imagery is deliberately NOT used here.
  roc_pr_d2.png       ROC and PR curves for the within-D2 endometrioma
                      result, from the stored out-of-fold predictions in
                      experiments/endomri_predictions.csv.
  threshold_sweep.gif animated decision-threshold sweep over the same
                      predictions (a "slider" that plays itself, since
                      GitHub READMEs cannot embed interactive controls).

Requires the UMD zip for the pipeline panel:  bash scripts/download_capi.sh
The ROC/PR figures only need the committed experiments/ CSV.

Usage:  python scripts/make_readme_figures.py [--skip-pipeline]
"""
import argparse
import os
import re
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SHAPEDEM_ROOT", os.path.join(REPO, "_workspace"))
OUT = os.path.join(REPO, "docs", "assets")

TEAL, INDIGO, CORAL, AMBER = "#0FA07F", "#6D69E0", "#E25A5A", "#D9A404"
# palette validated (dataviz six checks); amber's low white-surface contrast and
# the teal/coral deutan band are covered by direct per-structure text labels
GRAY, INK = "#868DA8", "#22304a"
SUBJECT = "UMD_221129_003"  # large corpus and fibroid; representative anatomy


# ---------------------------------------------------------------- pipeline fig
def pipeline_figure():
    import nibabel as nib
    import SimpleITK as sitk
    from skimage import measure

    zf = zipfile.ZipFile(os.path.join(ROOT, "data", "UMD.zip"))
    names = [n for n in zf.namelist() if "/._" not in n]
    segname = next(n for n in names if n.endswith(f"{SUBJECT}_seg.nii.gz"))
    dcms = sorted(n for n in names if f"/{SUBJECT}/" in n and n.endswith(".dcm"))

    with tempfile.TemporaryDirectory() as td:
        segp = os.path.join(td, "seg.nii.gz")
        open(segp, "wb").write(zf.read(segname))
        seg_img = nib.load(segp)
        seg = np.asanyarray(seg_img.dataobj)
        zooms = np.asarray(seg_img.header.get_zooms()[:3], float)

        ddir = os.path.join(td, "dcm"); os.mkdir(ddir)
        for n in dcms:
            open(os.path.join(ddir, os.path.basename(n)), "wb").write(zf.read(n))
        series = sitk.ImageSeriesReader()
        series.SetFileNames(series.GetGDCMSeriesFileNames(ddir))
        vol = sitk.GetArrayFromImage(series.Execute())  # (z, y, x)

    corpus = (seg == 1) | (seg == 2)
    fibroid = seg == 3
    # slice showing the most label types, then the most labeled area
    present = sum((seg == lab).any(axis=(0, 1)).astype(int) for lab in (1, 2, 3, 4))
    area = (seg > 0).sum(axis=(0, 1))
    z = int(np.argmax(present * 10**7 + area))
    sl = vol[z].T if vol.shape[1:] == seg.shape[:2] else vol[z]
    lo, hi = np.percentile(sl, [1, 99])
    sl = np.clip((sl - lo) / (hi - lo), 0, 1)
    sl = np.rot90(sl)  # spine vertical, anterior left: natural sagittal reading

    # raw marching cubes in mm, exactly as the pipeline computes it (no smoothing)
    verts, faces, _, _ = measure.marching_cubes(corpus.astype(np.uint8), 0.5, spacing=zooms)
    rng = np.random.default_rng(0)
    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(faces), 2048, p=areas / areas.sum())
    r1, r2 = rng.random((2, 2048))
    swap = r1 + r2 > 1
    r1[swap], r2[swap] = 1 - r1[swap], 1 - r2[swap]
    pts = (tri[pick, 0] * (1 - r1 - r2)[:, None]
           + tri[pick, 1] * r1[:, None] + tri[pick, 2] * r2[:, None])

    fig = plt.figure(figsize=(13, 3.6), facecolor="white")
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")

    aspect = zooms[1] / zooms[0]
    for ax, title in ((ax1, "a) T2 MRI slice"), (ax2, "b) + segmentation")):
        ax.imshow(sl, cmap="gray", aspect=1/aspect)
        ax.set_title(title, fontsize=11, color=INK)
        ax.axis("off")
    LABELS = ((1, "uterine wall", TEAL), (3, "fibroid", CORAL),
              (2, "cavity", INDIGO), (4, "cyst", AMBER))
    shown = 0
    for lab, name, color in LABELS:
        mask = np.rot90((seg[:, :, z] == lab).T)
        if not mask.any():
            continue
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = matplotlib.colors.to_rgba(color, 0.5)
        ax2.imshow(rgba, aspect=1/aspect)
        ax2.text(0.02, 0.02 + 0.07 * shown, name, color=color, fontsize=9,
                 transform=ax2.transAxes, fontweight="bold")
        shown += 1

    ax3.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                     color=TEAL, linewidth=0, antialiased=True, shade=True)
    ax3.set_title("c) marching-cubes mesh", fontsize=11, color=INK)
    ax4.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=INDIGO, alpha=0.8)
    ax4.set_title("d) 2,048-point cloud", fontsize=11, color=INK)
    for ax in (ax3, ax4):
        ax.view_init(elev=18, azim=-60)
        ax.set_box_aspect(np.ptp(verts, axis=0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.96, 0.96, 0.97, 1.0))
            axis.line.set_color(GRAY)
        ax.tick_params(colors=GRAY, labelsize=7, pad=-2)
        ax.set_xlabel("mm", fontsize=8, color=GRAY, labelpad=-6)
        ax.set_ylabel("mm", fontsize=8, color=GRAY, labelpad=-6)
        ax.set_zlabel("mm", fontsize=8, color=GRAY, labelpad=-6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipeline_umd.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[readme] pipeline_umd.png  (subject {SUBJECT}, slice {z})")


# ------------------------------------------------------------- ROC / PR + GIF
def load_d2():
    df = pd.read_csv(os.path.join(REPO, "experiments", "endomri_predictions.csv"))
    d2 = df[df.scope == "D2"]
    return d2.y_true.to_numpy(), d2.oof_prob.to_numpy()


def curve_axes(y, p):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    fpr, tpr, rth = roc_curve(y, p)
    prec, rec, pth = precision_recall_curve(y, p)
    return (fpr, tpr, rth, roc_auc_score(y, p)), (prec, rec, pth, average_precision_score(y, p))


def style(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def roc_pr_figure():
    y, p = load_d2()
    (fpr, tpr, _, auc), (prec, rec, _, ap) = curve_axes(y, p)
    prev = y.mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.1), facecolor="white")
    a1.plot(fpr, tpr, color=TEAL, lw=2)
    a1.plot([0, 1], [0, 1], color=GRAY, lw=1.4, ls="--")
    a1.text(0.55, 0.50, "random ranking (AUC 0.5)", color=GRAY, fontsize=9, rotation=38)
    a1.set_xlabel("false-positive rate", fontsize=10, color=INK)
    a1.set_ylabel("true-positive rate (sensitivity)", fontsize=10, color=INK)
    a1.set_title(f"ROC — within-site D2 (AUC {auc:.2f})", fontsize=11, color=INK)

    a2.plot(rec, prec, color=TEAL, lw=2)
    a2.axhline(prev, color=GRAY, lw=1.4, ls="--")
    a2.text(0.03, prev + 0.02, f"random ranking (= prevalence {prev:.2f})", color=GRAY, fontsize=9)
    a2.set_xlabel("recall (sensitivity)", fontsize=10, color=INK)
    a2.set_ylabel("precision", fontsize=10, color=INK)
    a2.set_ylim(0, 1.02)
    a2.set_title(f"Precision–recall (PR-AUC {ap:.2f})", fontsize=11, color=INK)
    for ax in (a1, a2):
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_pr_d2.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[readme] roc_pr_d2.png")


def threshold_gif():
    from matplotlib.animation import FuncAnimation, PillowWriter
    y, p = load_d2()
    (fpr, tpr, _, auc), (prec, rec, _, ap) = curve_axes(y, p)
    prev = y.mean()
    thresholds = np.quantile(p, np.linspace(0.02, 0.98, 36))[::-1]  # strict -> lenient

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4), facecolor="white")
    a1.plot(fpr, tpr, color=TEAL, lw=2)
    a1.plot([0, 1], [0, 1], color=GRAY, lw=1.2, ls="--")
    a2.plot(rec, prec, color=TEAL, lw=2)
    a2.axhline(prev, color=GRAY, lw=1.2, ls="--")
    a1.set_xlabel("false-positive rate", fontsize=10, color=INK)
    a1.set_ylabel("true-positive rate", fontsize=10, color=INK)
    a2.set_xlabel("recall", fontsize=10, color=INK)
    a2.set_ylabel("precision", fontsize=10, color=INK)
    a2.set_ylim(0, 1.02)
    a1.set_title(f"ROC (AUC {auc:.2f})", fontsize=11, color=INK)
    a2.set_title(f"Precision–recall (PR-AUC {ap:.2f})", fontsize=11, color=INK)
    for ax in (a1, a2):
        style(ax)
    d1 = a1.scatter([], [], s=70, color=INDIGO, zorder=5)
    d2 = a2.scatter([], [], s=70, color=INDIGO, zorder=5)
    txt = fig.suptitle("", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    def frame(t):
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        d1.set_offsets([[1 - spec, sens]])
        d2.set_offsets([[sens, ppv]])
        txt.set_text(f"decision threshold {t:.2f}   →   flags {tp + fp}/73 patients:   "
                     f"sensitivity {sens:.0%}   specificity {spec:.0%}   precision {ppv:.0%}")
        return d1, d2, txt

    anim = FuncAnimation(fig, frame, frames=thresholds, blit=False)
    anim.save(os.path.join(OUT, "threshold_sweep.gif"), writer=PillowWriter(fps=4), dpi=110)
    plt.close(fig)
    print("[readme] threshold_sweep.gif")


def trap_figure():
    import json
    e = json.load(open(os.path.join(REPO, "experiments", "endomri_results.json")))
    rows = [("Within site D2", e["within_D2_cv"]),
            ("Pooled 5-fold CV", e["endometrioma_cv"]),
            ("Within site D1", e["within_D1_cv"]),
            ("Transfer D1 \u2192 D2", e["site_D1_to_D2"])]
    names = [r[0] for r in rows]
    auc = [r[1]["auc"] for r in rows]
    ci = [r[1]["auc_ci95"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 3.4), facecolor="white")
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, auc, height=0.55, color=TEAL, zorder=3)
    ax.errorbar(auc, ypos,
                xerr=[[a - c[0] for a, c in zip(auc, ci)], [c[1] - a for a, c in zip(auc, ci)]],
                fmt="none", ecolor=INK, elinewidth=1.4, capsize=4, zorder=4)
    ax.axvline(0.5, color=GRAY, lw=1.4, ls="--", zorder=2)
    ax.text(0.5, len(rows) - 0.42, "random ranking (0.5)", color=GRAY, fontsize=9, ha="center")
    ax.set_ylim(-0.6, len(rows) - 0.2)
    for y, a in zip(ypos, auc):
        ax.text(0.02, y, f"{a:.2f}", va="center", color="white", fontsize=10, fontweight="bold", zorder=5)
    ax.set_yticks(ypos, names, fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC (whiskers: bootstrap 95% CI)", fontsize=10, color=INK)
    ax.set_title("The label trap at a glance: the same model, four evaluations", fontsize=11, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "label_trap.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[readme] label_trap.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="skip the panel that needs the UMD zip")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    threshold_gif()
    trap_figure()
    if not args.skip_pipeline:
        if os.path.exists(os.path.join(ROOT, "data", "UMD.zip")):
            pipeline_figure()
        else:
            sys.exit("UMD.zip not found; run scripts/download_capi.sh or use --skip-pipeline")
