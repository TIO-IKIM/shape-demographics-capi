"""Command-line entrypoints for the pipeline stages.

Two commands:
  paperb  -- run the CAPI-WOMEN paper pipeline (calls paperb.py):
             extract (NIfTI -> npz), analyze (XGBoost CV), tables (LaTeX output)
  smoke   -- end-to-end correctness gate on a tiny remote TotalSegmentator
             subset (calls pipeline.py + features.py + baselines.py directly)

Usage (env activated, repo on PYTHONPATH):
    python -m shapedem.cli paperb    [--step extract|analyze|tables|all]
    python -m shapedem.cli smoke
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import totalseg


def _meta(cfg, backend: str) -> pd.DataFrame:
    src = totalseg.from_config(cfg, backend)
    meta = src.read_meta()
    return totalseg.build_label_table(cfg, meta)


def _balanced_subjects(labels: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    df = labels.dropna(subset=["y_sex"])
    per = max(1, n // 2)
    parts = []
    for v in (0, 1):
        sub = df[df["y_sex"] == v]
        parts.append(sub.sample(min(per, len(sub)), random_state=seed))
    out = pd.concat(parts)
    return out["image_id"].tolist()


def cmd_paperb(args):
    from . import paperb
    cfg = load_config(args.config)
    if args.step in ("extract", "all"):
        paperb.extract_umd(cfg); paperb.extract_endo(cfg)
    if args.step in ("analyze", "all"):
        paperb.analyze_umd(cfg); paperb.analyze_endo(cfg); paperb.rater_agreement(cfg)
    if args.step in ("tables", "all"):
        paperb.write_paperb_tables(cfg)


def cmd_smoke(args):
    """End-to-end correctness gate on a tiny remote subset."""
    from .pipeline import extract_subject
    from .features import assemble_descriptors, select_columns, organ_presence
    from .train.baselines import cv_classification, cv_regression
    cfg = load_config(args.config)
    labels = _meta(cfg, "remote")
    subjects = _balanced_subjects(labels, cfg["smoke"]["n_subjects"])
    organs = cfg["smoke"]["organs"]
    print(f"[smoke] {len(subjects)} subjects, organs={organs}")
    src = totalseg.from_config(cfg, "remote")
    with src:
        for i, subj in enumerate(subjects):
            st = extract_subject(cfg, src, "totalseg", subj, organs)
            print(f"[smoke] {i+1}/{len(subjects)} {subj}: {st}")
    pres = organ_presence(cfg, "totalseg", subjects, organs)
    print("[smoke] presence:\n", pres.to_string())
    X = assemble_descriptors(cfg, "totalseg", subjects, organs)
    lab = labels.set_index("image_id").reindex(subjects)
    full = X[select_columns(X, "full")]
    res = {
        "n_subjects": len(subjects), "n_features": full.shape[1],
        "presence": pres.to_dict(),
        "sex_full": cv_classification(full, lab["y_sex"].to_numpy(), kind="logreg", n_splits=3),
        "age_full": cv_regression(full, lab["y_age"].to_numpy(), kind="ridge", n_splits=3),
    }
    print(json.dumps(res, indent=2))
    out = os.path.join(cfg["paths"]["results_dir"], "smoke_results.json")
    json.dump(res, open(out, "w"), indent=2)
    # sanity assertions
    assert full.shape[1] > 0, "no features produced"
    assert pres.max() > 0, "no organ extracted for any subject"
    print(f"[smoke] OK -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser("shapedem")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("paperb")
    pb.add_argument("--step", default="all", choices=["extract", "analyze", "tables", "all"])
    pb.set_defaults(fn=cmd_paperb)
    s = sub.add_parser("smoke"); s.set_defaults(fn=cmd_smoke)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
