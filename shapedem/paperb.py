"""CAPI-WOMEN: shape-based phenotyping of female pelvic organs.

Entry point: cli.py calls extract_umd/extract_endo, analyze_umd/analyze_endo,
rater_agreement, and write_paperb_tables in sequence (--step extract/analyze/tables).

Flow:
  extract_*  -> opens dataset zip, extracts NIfTI masks, calls shapes/extract.py
                to produce npz point clouds, writes labels CSV to experiments/
  analyze_*  -> loads npz via features.py, builds descriptor DataFrame, calls
                train/baselines.py for XGBoost CV, writes results JSON to experiments/
  rater_agreement -> computes inter-rater CoV from multiple segmentations
  write_paperb_tables -> reads result JSONs, emits LaTeX tables + macros + figure
                         to experiments/ (these are \\input'd by the paper)

Datasets:
  * UMD  (uterine myoma MRI, 300 women): uterus shape -> AGE regression
         (age read from DICOM PatientAge), plus myoma burden.
  * UT-EndoMRI (124 usable segmentations, 2 sites): ovary/uterus shape ->
         ENDOMETRIOMA presence, with D1->D2 multi-site transfer.
"""
from __future__ import annotations
import os
import re
import tempfile
import zipfile
import json

import numpy as np
import pandas as pd

from .config import load_config
from .shapes.extract import labelmask_to_pointcloud, mask_to_pointcloud, save_shape
from .pipeline import feature_path

# pelvic MRI: the organ is the scan target, so don't apply FOV-truncation rejection
_BTOL = 10 ** 12
UMD_LABELS = {"uterus": [1, 2], "uterus_total": [1, 2, 3], "myoma": [3]}


def _voxel_spacing(path):
    """Voxel spacing (mm) of a NIfTI file, for reporting dataset resolution."""
    import nibabel as nib
    try:
        z = nib.load(path).header.get_zooms()[:3]
        return [float(v) for v in z]
    except Exception:
        return [float("nan")] * 3


# ----------------------------- UMD -----------------------------
def _umd_zip(cfg):
    return os.path.join(cfg["paths"]["data_dir"], "UMD.zip")


def umd_subjects(zf):
    return sorted({n.split("/")[1] for n in zf.namelist()
                   if n.startswith("UMD/UMD_") and "/._" not in n and len(n.split("/")) > 1})


def umd_age(zf, subj, tmp):
    import SimpleITK as sitk
    dcms = [n for n in zf.namelist() if f"/{subj}/" in n and n.endswith(".dcm") and "/._" not in n]
    for m in dcms[:1]:
        p = os.path.join(tmp, "a.dcm"); open(p, "wb").write(zf.read(m))
        r = sitk.ImageFileReader(); r.SetFileName(p); r.ReadImageInformation()
        try:
            raw = r.GetMetaData("0010|1010").strip()  # e.g. '044Y'
            mt = re.match(r"(\d+)", raw)
            if mt:
                return float(mt.group(1))
        except Exception:
            pass
    return float("nan")


def extract_umd(cfg, overwrite=False):
    zf = zipfile.ZipFile(_umd_zip(cfg))
    subjects = umd_subjects(zf)
    rows = []
    for i, subj in enumerate(subjects):
        seg = [n for n in zf.namelist() if n.endswith(f"{subj}_seg.nii.gz") and "/._" not in n]
        with tempfile.TemporaryDirectory(dir=cfg["paths"]["work_dir"]) as tmp:
            age = umd_age(zf, subj, tmp)
            rows.append({"image_id": subj, "y_age": age, "y_sex": 1.0})  # all female
            if not seg:
                continue
            sp = os.path.join(tmp, "seg.nii.gz"); open(sp, "wb").write(zf.read(seg[0]))
            vs = _voxel_spacing(sp)
            rows[-1].update({"spacing_x": vs[0], "spacing_y": vs[1], "spacing_z": vs[2]})
            for struct, labels in UMD_LABELS.items():
                fp = feature_path(cfg, "umd", subj, struct)
                if os.path.exists(fp) and not overwrite:
                    continue
                try:
                    sh = labelmask_to_pointcloud(sp, labels, min_voxels=50, boundary_tol=_BTOL)
                except Exception:
                    sh = None
                if sh is not None:
                    save_shape(fp, sh)
        if (i + 1) % 50 == 0:
            print(f"[umd] {i+1}/{len(subjects)}")
    lab = pd.DataFrame(rows)
    lab.to_csv(os.path.join(cfg["paths"]["results_dir"], "umd_labels.csv"), index=False)
    print(f"[umd] {len(lab)} subjects, age available for {lab['y_age'].notna().sum()}")
    return lab


# ----------------------------- UT-EndoMRI -----------------------------
def _endo_zip(cfg):
    return os.path.join(cfg["paths"]["data_dir"], "UT-EndoMRI.zip")


def endo_subjects(zf):
    out = []
    for n in zf.namelist():
        m = re.match(r"UT-EndoMRI/(D[12]_[A-Z]+)/(D[12]-\d+)/", n)
        if m and "/._" not in n:
            out.append((m.group(1), m.group(2)))
    return sorted(set(out))


def _best_struct_file(zf, subj, struct):
    """Pick the structure mask, preferring rater r3>r2>r1>unsuffixed; return member or None."""
    cands = [n for n in zf.namelist()
             if re.search(rf"/{subj}_{struct}(_r\d+)?\.nii\.gz$", n) and "/._" not in n]
    if not cands:
        return None
    def rank(n):
        m = re.search(r"_r(\d+)\.nii\.gz$", n)
        return int(m.group(1)) if m else 0
    return sorted(cands, key=rank, reverse=True)[0]


def extract_endo(cfg, overwrite=False):
    zf = zipfile.ZipFile(_endo_zip(cfg))
    subs = endo_subjects(zf)
    rows = []
    for i, (inst, subj) in enumerate(subs):
        em = _best_struct_file(zf, subj, "em")
        rows.append({"image_id": subj, "institute": inst,
                     "y_endometrioma": 1.0 if em else 0.0, "y_sex": 1.0})
        with tempfile.TemporaryDirectory(dir=cfg["paths"]["work_dir"]) as tmp:
            for struct in ("ut", "ov"):
                member = _best_struct_file(zf, subj, struct)
                if not member:
                    continue
                fp = feature_path(cfg, "endomri", subj, {"ut": "uterus", "ov": "ovary"}[struct])
                if os.path.exists(fp) and not overwrite:
                    continue
                p = os.path.join(tmp, "m.nii.gz"); open(p, "wb").write(zf.read(member))
                if "spacing_x" not in rows[-1]:
                    vs = _voxel_spacing(p)
                    rows[-1].update({"spacing_x": vs[0], "spacing_y": vs[1], "spacing_z": vs[2]})
                try:
                    sh = mask_to_pointcloud(p, min_voxels=30, boundary_tol=_BTOL)
                except Exception:
                    sh = None
                if sh is not None:
                    save_shape(fp, sh)
        if (i + 1) % 30 == 0:
            print(f"[endo] {i+1}/{len(subs)}")
    lab = pd.DataFrame(rows)
    lab.to_csv(os.path.join(cfg["paths"]["results_dir"], "endomri_labels.csv"), index=False)
    print(f"[endo] {len(lab)} subjects; endometrioma+ = {int(lab['y_endometrioma'].sum())}; "
          f"sites = {lab['institute'].value_counts().to_dict()}")
    return lab


# ----------------------------- output generation -----------------------------
def write_paperb_tables(cfg):
    """Read result JSONs from experiments/, emit LaTeX tables + macros + figure.

    Outputs (all under experiments/):
      tables/umd_age.tex  -- Table 1: age prediction by feature regime
      tables/endo.tex     -- Table 2: endometrioma detection (pooled/within/cross-site)
      tables/rater.tex    -- Table 3: inter-rater CoV (mean + median)
      results_macros.tex  -- \\newcommand macros for every number cited in the paper
      figures/umd_age.*   -- bar chart of age MAE by feature regime
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    rdir = cfg["paths"]["results_dir"]
    tdir = os.path.join(rdir, "tables"); fdir = os.path.join(rdir, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    umd = json.load(open(os.path.join(rdir, "umd_results.json")))
    endo = json.load(open(os.path.join(rdir, "endomri_results.json")))
    umdlab = pd.read_csv(os.path.join(rdir, "umd_labels.csv"))

    rater_csv = os.path.join(rdir, "rater_agreement.csv")
    rad = pd.read_csv(rater_csv) if os.path.exists(rater_csv) else None

    def _pm(d, key, std, fmt):
        s = d.get(std)
        return fmt.format(d[key]) + (f"\\,$\\pm$\\,{fmt.format(s)}" if isinstance(s, (int, float)) and s == s else "")

    # UMD age table: feature regimes + per-structure (uterus corpus vs fibroid), with 5-fold std
    with open(os.path.join(tdir, "umd_age.tex"), "w") as f:
        f.write("\\begin{tabular}{lcc}\n\\toprule\nFeatures & Age MAE (yr) & Age $R^2$ \\\\\n\\midrule\n")
        for reg, name in (("size", "Size only"), ("shape", "Shape only"), ("full", "Size $+$ shape")):
            r = umd[f"age_{reg}"]
            f.write(f"{name} & {_pm(r,'mae','mae_std','{:.1f}')} & {_pm(r,'r2','r2_std','{:.2f}')} \\\\\n")
        f.write("\\midrule\n")
        for st, name in (("uterus", "Uterine corpus shape"), ("myoma", "Fibroid shape")):
            r = umd[f"age_struct_{st}"]
            f.write(f"{name} & {_pm(r,'mae','mae_std','{:.1f}')} & {_pm(r,'r2','r2_std','{:.2f}')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # endo table: pooled, within-site, cross-site (with prevalence + std; D1 is uninformative).
    # ROC-AUC carries the fold std; PR-AUC and balanced accuracy contextualize the imbalance.
    pv = endo["endometrioma_prevalence_by_site"]
    d1p = 100 * next((v for k, v in pv.items() if k.startswith("D1")), float("nan"))
    d2p = 100 * next((v for k, v in pv.items() if k.startswith("D2")), float("nan"))

    def _opt(d, key, fmt="{:.3f}"):
        v = d.get(key)
        return fmt.format(v) if isinstance(v, (int, float)) and v == v else "--"

    with open(os.path.join(tdir, "endo.tex"), "w") as f:
        f.write("\\begin{tabular}{lrccc}\n\\toprule\n"
                "Setting & $n$ & ROC-AUC & PR-AUC & Bal.\\ acc. \\\\\n\\midrule\n")
        f.write(f"Pooled (5-fold CV) & {endo['n']} & "
                f"{_pm(endo['endometrioma_cv'],'auc','auc_std','{:.3f}')} & "
                f"{_opt(endo['endometrioma_cv'],'pr_auc')} & "
                f"{_pm(endo['endometrioma_cv'],'balanced_acc','balanced_acc_std','{:.2f}')} \\\\\n")
        if "within_D2_cv" in endo:
            f.write(f"Within site D2 ({d2p:.0f}\\% pos.) & {endo['within_D2_cv']['n']} & "
                    f"{_pm(endo['within_D2_cv'],'auc','auc_std','{:.3f}')} & "
                    f"{_opt(endo['within_D2_cv'],'pr_auc')} & "
                    f"{_pm(endo['within_D2_cv'],'balanced_acc','balanced_acc_std','{:.2f}')} \\\\\n")
        if "within_D1_cv" in endo:
            f.write(f"Within site D1 ({d1p:.0f}\\% pos.) & {endo['within_D1_cv']['n']} & "
                    f"{_pm(endo['within_D1_cv'],'auc','auc_std','{:.3f}')} & "
                    f"{_opt(endo['within_D1_cv'],'pr_auc')} & "
                    f"{_pm(endo['within_D1_cv'],'balanced_acc','balanced_acc_std','{:.2f}')} \\\\\n")
        f.write(f"Cross-site D1$\\rightarrow$D2 & {endo['site_D1_to_D2']['n_test']} & "
                f"{endo['site_D1_to_D2']['auc']:.3f} & "
                f"{_opt(endo['site_D1_to_D2'],'pr_auc')} & "
                f"{_opt(endo['site_D1_to_D2'],'balanced_acc','{:.2f}')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # rater agreement table (+ subject counts)
    raw = pd.read_csv(os.path.join(rdir, "rater_cov_raw.csv")) if os.path.exists(os.path.join(rdir, "rater_cov_raw.csv")) else None
    nsub = raw.groupby("struct")["subject"].nunique().to_dict() if raw is not None else {}
    with open(os.path.join(tdir, "rater.tex"), "w") as f:
        if rad is None:
            f.write("% run `paperb --step analyze`\n")
        else:
            f.write("\\begin{tabular}{llrcc}\n\\toprule\nOrgan & Feature family & subjects & Mean CoV & Median CoV \\\\\n\\midrule\n")
            disp = {"ut": "Uterus", "ov": "Ovary"}
            for _, r in rad.iterrows():
                f.write(f"{disp.get(r['struct'], r['struct'])} & {r['family']} & "
                        f"{int(nsub.get(r['struct'], 0))} & {r['mean']:.3f} & {r['median']:.3f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}\n"
    def rget(st, fam):
        if rad is None:
            return float("nan")
        m = rad[(rad.struct == st) & (rad.family == fam)]
        return float(m["mean"].iloc[0]) if len(m) else float("nan")
    def rgetmed(st, fam):
        if rad is None:
            return float("nan")
        m = rad[(rad.struct == st) & (rad.family == fam)]
        return float(m["median"].iloc[0]) if len(m) else float("nan")
    d1 = next((v for k, v in pv.items() if k.startswith("D1")), float("nan"))
    d2 = next((v for k, v in pv.items() if k.startswith("D2")), float("nan"))

    # camera-ready context numbers, all derived from the same experiment files
    endolab = pd.read_csv(os.path.join(rdir, "endomri_labels.csv"))
    age = umdlab["y_age"].dropna()
    base_mae = float((age - age.mean()).abs().mean())
    ut_mae = umd["age_struct_uterus"]["mae"]
    ut_r2 = umd["age_struct_uterus"]["r2"]

    def _spacing_macro(df):
        cols = ["spacing_x", "spacing_y", "spacing_z"]
        if not all(c in df.columns for c in cols):
            return "[[TODO: spacing not captured; re-run extract]]"
        med = df[cols].median()
        return f"{med.iloc[0]:.1f}\\times{med.iloc[1]:.1f}\\times{med.iloc[2]:.1f}"

    def _ci_macro(d):
        ci = d.get("auc_ci95")
        return f"{ci[0]:.2f}--{ci[1]:.2f}" if ci else "[[TODO: CI missing; re-run analyze]]"

    npos = endolab.groupby("institute")["y_endometrioma"].agg(["sum", "count"])
    d1pos = int(npos.loc["D1_MHS", "sum"]); d1n = int(npos.loc["D1_MHS", "count"])
    d2pos = int(npos.loc["D2_TCPW", "sum"])
    covmed = (raw[raw.struct == "ut"].groupby(["family", "feature"])["cov"].median()
              if raw is not None else None)
    covnum = covmed[covmed.index.get_level_values("family").isin(["size", "shape"])] if covmed is not None else None
    d2cv = endo.get("within_D2_cv", {})
    macros = "".join([
        mac("UMDn", f"{umd['n']}"),
        mac("UMDageMAE", f"{umd['age_full']['mae']:.1f}"),
        mac("UMDageRtwo", f"{umd['age_full']['r2']:.2f}"),
        mac("UMDageMAEsize", f"{umd['age_size']['mae']:.1f}"),
        mac("UMDuterusRtwo", f"{umd['age_struct_uterus']['r2']:.2f}"),
        mac("UMDmyomaRtwo", f"{umd['age_struct_myoma']['r2']:.2f}"),
        mac("UMDageMin", f"{umdlab['y_age'].min():.0f}"),
        mac("UMDageMax", f"{umdlab['y_age'].max():.0f}"),
        mac("UMDageMean", f"{umdlab['y_age'].mean():.0f}"),
        mac("EndoN", f"{endo['n']}"),
        mac("EndoCVauc", f"{endo['endometrioma_cv']['auc']:.3f}"),
        mac("EndoDtwoAUC", f"{endo.get('within_D2_cv', {}).get('auc', float('nan')):.3f}"),
        mac("EndoDtwoAUCstd", f"{endo.get('within_D2_cv', {}).get('auc_std', float('nan')):.3f}"),
        mac("EndoDoneAUC", f"{endo.get('within_D1_cv', {}).get('auc', float('nan')):.3f}"),
        mac("EndoTransferAUC", f"{endo['site_D1_to_D2']['auc']:.3f}"),
        mac("EndoTransferN", f"{endo['site_D1_to_D2']['n_test']}"),
        mac("EndoDoneprev", f"{100*d1:.0f}"),
        mac("EndoDtwoprev", f"{100*d2:.0f}"),
        mac("UMDuterusMAE", f"{umd['age_struct_uterus']['mae']:.1f}"),
        mac("UMDageSizeRtwo", f"{umd['age_size']['r2']:.2f}"),
        mac("UMDageShapeRtwo", f"{umd['age_shape']['r2']:.2f}"),
        mac("RaterUtShapeCoV", f"{rget('ut','shape'):.3f}"),
        mac("RaterUtSizeCoV", f"{rget('ut','size'):.3f}"),
        mac("RaterOvShapeCoV", f"{rget('ov','shape'):.3f}"),
        mac("RaterOvSizeCoV", f"{rget('ov','size'):.3f}"),
        mac("RaterNut", f"{int(nsub.get('ut', 0))}"),
        mac("RaterNov", f"{int(nsub.get('ov', 0))}"),
        mac("RaterUtShapeCoVmed", f"{rgetmed('ut','shape'):.3f}"),
        mac("RaterUtSizeCoVmed", f"{rgetmed('ut','size'):.3f}"),
        mac("RaterOvShapeCoVmed", f"{rgetmed('ov','shape'):.3f}"),
        mac("RaterOvSizeCoVmed", f"{rgetmed('ov','size'):.3f}"),
        # camera-ready additions (context, counts, uncertainty)
        mac("UMDageSD", f"{age.std():.1f}"),
        mac("UMDageBaseMAE", f"{base_mae:.1f}"),
        mac("UMDageMAEredpct", f"{100 * (1 - ut_mae / base_mae):.0f}"),
        mac("UMDuterusRtwopct", f"{100 * ut_r2:.0f}"),
        mac("UMDspacing", _spacing_macro(umdlab)),
        mac("EndoSpacing", _spacing_macro(endolab)),
        mac("UMDnUterus", f"{round(umd['presence']['uterus'] * umd['n'])}"),
        mac("UMDnMyoma", f"{round(umd['presence']['myoma'] * umd['n'])}"),
        mac("EndoNut", f"{round(endo['presence']['uterus'] * endo['n'])}"),
        mac("EndoNov", f"{round(endo['presence']['ovary'] * endo['n'])}"),
        mac("EndoDonePosN", f"{d1pos}"),
        mac("EndoDoneNegN", f"{d1n - d1pos}"),
        mac("EndoDtwoPosN", f"{d2pos}"),
        mac("EndoDtwoPrevFrac", f"{d2:.2f}"),
        mac("EndoDtwoAUCci", _ci_macro(d2cv)),
        mac("EndoDoneAUCci", _ci_macro(endo.get("within_D1_cv", {}))),
        mac("EndoDoneAUCstd", f"{endo.get('within_D1_cv', {}).get('auc_std', float('nan')):.3f}"),
        mac("EndoDtwoPRauc", f"{d2cv.get('pr_auc', float('nan')):.3f}"),
        mac("EndoDtwoBacc", f"{d2cv.get('balanced_acc', float('nan')):.2f}"),
        mac("EndoDtwoBaccStd", f"{d2cv.get('balanced_acc_std', float('nan')):.2f}"),
        mac("EndoDtwoPermP", f"{d2cv.get('perm_p', float('nan')):.3f}"),
        mac("RaterUtBestCoV", f"{covnum.min():.3f}" if covnum is not None else "nan"),
        mac("RaterUtWorstCoV", f"{covnum.max():.3f}" if covnum is not None else "nan"),
        mac("RaterUtWorstShapeCoV",
            f"{covmed.loc['shape'].max():.3f}" if covmed is not None else "nan"),
    ])
    open(os.path.join(rdir, "results_macros.tex"), "w").write(
        "% AUTO-GENERATED from experiments/ — do not edit.\n" + macros)

    # figure: UMD age MAE by regime, against the cohort-mean baseline
    fig, ax = plt.subplots(figsize=(4.2, 3))
    regs = ["size", "shape", "full"]
    ax.bar([r.capitalize() for r in regs], [umd[f"age_{r}"]["mae"] for r in regs], color="#b0539a")
    ax.axhline(base_mae, color="#333333", linestyle="--", linewidth=1)
    ax.text(0.98, base_mae, f"cohort-mean baseline ({base_mae:.1f} yr)",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    ax.set_ylabel("Uterine-shape age MAE (yr)"); ax.set_title("Age from uterine morphology")
    fig.savefig(os.path.join(fdir, "umd_age.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(fdir, "umd_age.pdf"), bbox_inches="tight"); plt.close(fig)
    print(f"[capi] tables/macros/figure -> {rdir}")


def analyze_umd(cfg):
    from .features import assemble_descriptors, select_columns, organ_presence
    from .train.baselines import cv_regression
    rdir = cfg["paths"]["results_dir"]
    lab = pd.read_csv(os.path.join(rdir, "umd_labels.csv")).set_index("image_id")
    subjects = lab.index.tolist()
    organs = list(UMD_LABELS.keys())
    pres = organ_presence(cfg, "umd", subjects, organs)
    X = assemble_descriptors(cfg, "umd", subjects, organs)
    y = lab.reindex(X.index)
    out = {"n": int(len(X)), "presence": pres.to_dict()}
    for regime in ("size", "shape", "full"):
        cols = [c for c in select_columns(X, regime) if not c.endswith("__present")]
        out[f"age_{regime}"] = cv_regression(X[cols], y["y_age"].to_numpy(), kind="xgb")
    # is the age signal just fibroid burden? per-structure age regression
    for st in ("uterus", "myoma", "uterus_total"):
        cols = [c for c in X.columns if c.startswith(st + "__") and not c.endswith("__present")]
        out[f"age_struct_{st}"] = cv_regression(X[cols], y["y_age"].to_numpy(), kind="xgb")
    json.dump(out, open(os.path.join(rdir, "umd_results.json"), "w"), indent=2)
    print("[umd]", json.dumps(out, indent=2))
    return out


def analyze_endo(cfg):
    from .features import assemble_descriptors, organ_presence
    from .train.baselines import cv_classification, fit_eval_classification
    rdir = cfg["paths"]["results_dir"]
    lab = pd.read_csv(os.path.join(rdir, "endomri_labels.csv")).set_index("image_id")
    subjects = lab.index.tolist()
    organs = ["uterus", "ovary"]
    pres = organ_presence(cfg, "endomri", subjects, organs)
    X = assemble_descriptors(cfg, "endomri", subjects, organs)
    y = lab.reindex(X.index)
    cols = [c for c in X.columns if not c.endswith("__present")]
    # endometrioma-label prevalence by site (reveals annotation/site confound)
    prev = (y.groupby("institute")["y_endometrioma"].mean()).to_dict()
    out = {"n": int(len(X)), "presence": pres.to_dict(),
           "endometrioma_prevalence_by_site": prev,
           "endometrioma_cv": cv_classification(X[cols], y["y_endometrioma"].to_numpy(), kind="xgb")}
    # multi-site: train D1 -> test D2
    d1 = y["institute"] == "D1_MHS"; d2 = y["institute"] == "D2_TCPW"
    out["site_D1_to_D2"], yte, proba = fit_eval_classification(
        X.loc[d1, cols], y.loc[d1, "y_endometrioma"].to_numpy(),
        X.loc[d2, cols], y.loc[d2, "y_endometrioma"].to_numpy(), kind="xgb",
        return_proba=True)
    # within-site CV (the honest analysis given the prevalence confound)
    for site, mask in (("D1", d1), ("D2", d2)):
        yy = y.loc[mask, "y_endometrioma"].to_numpy()
        if 0 < yy.sum() < len(yy):  # both classes present
            out[f"within_{site}_cv"] = cv_classification(X.loc[mask, cols], yy, kind="xgb")
    _endo_uncertainty(out, X, y, cols, d1, d2, yte, proba, rdir)
    json.dump(out, open(os.path.join(rdir, "endomri_results.json"), "w"), indent=2)
    print("[endo]", json.dumps(out, indent=2))
    return out


def _endo_uncertainty(out, X, y, cols, d1, d2, yte, proba, rdir):
    """Bootstrap CIs, PR-AUC, and a permutation test on pooled OOF predictions.

    Uses the same folds, seed, and rebalancing as cv_classification, so the
    fold-level numbers in `out` are untouched; this only adds uncertainty
    estimates and persists the out-of-fold predictions for reproducibility.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from .train.baselines import oof_classification, bootstrap_ci, permutation_pvalue
    pred_rows = []
    for key, mask in (("endometrioma_cv", None), ("within_D1_cv", d1), ("within_D2_cv", d2)):
        if key not in out:
            continue
        Xs = X[cols] if mask is None else X.loc[mask, cols]
        ys = (y if mask is None else y.loc[mask])["y_endometrioma"].to_numpy()
        yt, prob = oof_classification(Xs, ys, kind="xgb")
        _, lo, hi = bootstrap_ci(yt, prob, roc_auc_score)
        out[key]["auc_ci95"] = [round(lo, 3), round(hi, 3)]
        out[key]["pr_auc"] = float(average_precision_score(yt, prob))
        scope = "pooled" if mask is None else key.split("_")[1]
        pred_rows += [{"scope": scope, "image_id": i, "y_true": int(t), "oof_prob": float(p)}
                      for i, t, p in zip(Xs.index, yt, prob)]
    # permutation test for the headline within-D2 result
    if "within_D2_cv" in out:
        p, obs = permutation_pvalue(X.loc[d2, cols], y.loc[d2, "y_endometrioma"].to_numpy(),
                                    kind="xgb", n_perm=1000)
        out["within_D2_cv"]["perm_p"] = p
        out["within_D2_cv"]["oof_auc"] = round(obs, 3)
    # transfer: bootstrap over the D2 test predictions of the single D1-trained model
    if len(proba):
        _, lo, hi = bootstrap_ci(yte, proba, roc_auc_score)
        out["site_D1_to_D2"]["auc_ci95"] = [round(lo, 3), round(hi, 3)]
        out["site_D1_to_D2"]["pr_auc"] = float(average_precision_score(yte, proba))
        pred_rows += [{"scope": "transfer_D1_to_D2", "image_id": i, "y_true": int(t),
                       "oof_prob": float(p)}
                      for i, t, p in zip(X.loc[d2].index, yte, proba)]
    pd.DataFrame(pred_rows).to_csv(os.path.join(rdir, "endomri_predictions.csv"), index=False)


def rater_agreement(cfg):
    """Inter-rater shape robustness: how much do shape descriptors vary across the
    3 raters' segmentations of the same D1 organ? Reports mean coefficient of
    variation (CoV) across raters, split into size vs scale-free shape features."""
    import zipfile
    import tempfile
    import re
    from .shapes.extract import mask_to_pointcloud
    from .shapes.descriptors import shape_descriptors, SIZE_PREFIX, SHAPE_PREFIX
    rdir = cfg["paths"]["results_dir"]
    zf = zipfile.ZipFile(_endo_zip(cfg))
    names = [n for n in zf.namelist() if "/._" not in n and n.endswith(".nii.gz")]
    d1subs = sorted({m.group(1) for n in names
                     if (m := re.search(r"/(D1-\d+)/", n))})
    rows = []
    for struct in ("ut", "ov"):
        for subj in d1subs:
            rmasks = {}
            for r in ("r1", "r2", "r3"):
                c = [n for n in names if n.endswith(f"/{subj}_{struct}_{r}.nii.gz")]
                if c:
                    rmasks[r] = c[0]
            if len(rmasks) < 2:
                continue
            descs = {}
            with tempfile.TemporaryDirectory(dir=cfg["paths"]["work_dir"]) as tmp:
                for r, member in rmasks.items():
                    p = os.path.join(tmp, "m.nii.gz"); open(p, "wb").write(zf.read(member))
                    sh = mask_to_pointcloud(p, min_voxels=30, boundary_tol=10 ** 12)
                    if sh is not None:
                        descs[r] = shape_descriptors(sh)
            if len(descs) < 2:
                continue
            keys = sorted(set.intersection(*[set(d) for d in descs.values()]))
            for k in keys:
                vals = np.array([descs[r][k] for r in descs], float)
                if np.all(np.isfinite(vals)) and abs(vals.mean()) > 1e-9:
                    fam = ("size" if k.startswith(SIZE_PREFIX)
                           else "shape" if k.startswith(SHAPE_PREFIX) else "other")
                    rows.append({"struct": struct, "subject": subj, "feature": k,
                                 "family": fam, "cov": float(vals.std() / abs(vals.mean())),
                                 "n_raters": len(descs)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rdir, "rater_cov_raw.csv"), index=False)
    summ = (df[df.family.isin(["size", "shape"])]
            .groupby(["struct", "family"])["cov"].agg(["mean", "median", "count"]).reset_index())
    summ.to_csv(os.path.join(rdir, "rater_agreement.csv"), index=False)
    n_subj = df.groupby("struct")["subject"].nunique().to_dict()
    print(f"[rater] multi-rater subjects per struct: {n_subj}")
    print("[rater] mean CoV by struct/family:\n", summ.to_string(index=False))
    return summ
