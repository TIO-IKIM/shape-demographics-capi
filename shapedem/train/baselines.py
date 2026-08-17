"""Classical (correspondence-free) baselines + cross-validated evaluation.

Called by paperb.py (CAPI-WOMEN analysis) and cli.py (smoke test). Provides:
  - cv_classification / cv_regression: stratified K-fold CV with fold-level stats
  - fit_eval_classification / fit_eval_regression: single train/test split (cross-site)
  - oof_classification / oof_regression + bootstrap_ci: pooled OOF predictions
    with bootstrap confidence intervals (available but not used by paperb.py)

XGBoost classifiers use per-fold scale_pos_weight for class rebalancing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mean_absolute_error, r2_score


def _clf(kind: str):
    if kind == "logreg":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    if kind == "xgb":
        # XGBoost handles NaN natively (learns split direction for missing
        # values), so no imputer is needed — unlike LogReg/Ridge which require
        # median imputation via a sklearn Pipeline.
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", n_jobs=8,
        )
    raise ValueError(kind)


def _reg(kind: str):
    if kind == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=10.0)),
        ])
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=8,
        )
    raise ValueError(kind)


def oof_classification(X, y, kind="xgb", n_splits=5, seed=0):
    """Out-of-fold predicted probabilities (for bootstrap CIs).

    Uses the same per-fold class rebalancing (scale_pos_weight) as
    cv_classification, so the pooled OOF predictions come from the same
    model family as the fold-level metrics.
    """
    y = np.asarray(y); m = ~np.isnan(y)
    X, y = X.loc[m], y[m].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        mdl = _clf(kind)
        if hasattr(mdl, "scale_pos_weight"):
            mdl.set_params(scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        mdl.fit(X.iloc[tr], y[tr])
        prob[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    return y, prob


def oof_regression(X, y, kind="xgb", n_splits=5, seed=0):
    y = np.asarray(y, float); m = ~np.isnan(y)
    X, y = X.loc[m], y[m]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        mdl = _reg(kind); mdl.fit(X.iloc[tr], y[tr]); pred[te] = mdl.predict(X.iloc[te])
    return y, pred


def bootstrap_ci(y_true, y_score, metric, n_boot=2000, seed=0):
    """Percentile bootstrap CI of a metric over (y_true, y_score) resamples.

    Resamples where the metric is undefined (raises, or returns NaN as
    sklearn does for single-class y_true) are skipped, not propagated.
    """
    rng = np.random.RandomState(seed); n = len(y_true); vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            v = float(metric(y_true[idx], y_score[idx]))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def permutation_pvalue(X, y, kind="xgb", n_perm=1000, n_splits=5, seed=0):
    """Label-permutation p-value for the pooled OOF ROC-AUC.

    Repeats the full OOF cross-validation with permuted labels; the p-value is
    the fraction of permutations whose AUC reaches the observed one, with the
    +1 correction of Phipson & Smyth so p is never exactly zero.
    """
    from sklearn.metrics import roc_auc_score
    yt, prob = oof_classification(X, y, kind=kind, n_splits=n_splits, seed=seed)
    obs = roc_auc_score(yt, prob)
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(n_perm):
        yp = rng.permutation(yt)
        pt, pp = oof_classification(X, yp.astype(float), kind=kind, n_splits=n_splits, seed=seed)
        if roc_auc_score(pt, pp) >= obs:
            hits += 1
    return float((hits + 1) / (n_perm + 1)), float(obs)


def cv_classification(X: pd.DataFrame, y: np.ndarray, kind: str = "xgb",
                      n_splits: int = 5, seed: int = 0) -> dict:
    y = np.asarray(y)
    m = ~np.isnan(y)
    X, y = X.loc[m], y[m].astype(int)
    n_splits = min(n_splits, np.bincount(y).min()) if len(np.unique(y)) > 1 else 2
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        model = _clf(kind)
        if hasattr(model, "scale_pos_weight"):
            model.set_params(scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(balanced_accuracy_score(y[te], pred))
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], proba))
    return {"n": int(m.sum()), "balanced_acc": float(np.mean(accs)),
            "balanced_acc_std": float(np.std(accs)),
            "auc": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std": float(np.std(aucs)) if aucs else float("nan")}


def fit_eval_classification(Xtr, ytr, Xte, yte, kind: str = "xgb",
                            return_proba: bool = False):
    """Train on one split, evaluate on another (used for cross-domain transfer).

    With return_proba=True, also returns (y_test, proba) so callers can compute
    additional metrics (e.g., bootstrap CIs) on the test predictions.
    """
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
    Xtr, ytr = Xtr.loc[mtr], ytr[mtr].astype(int)
    Xte, yte = Xte.loc[mte], yte[mte].astype(int)
    if len(np.unique(ytr)) < 2 or len(Xte) == 0:
        res = {"n_train": int(mtr.sum()), "n_test": int(mte.sum()),
               "balanced_acc": float("nan"), "auc": float("nan")}
        return (res, yte, np.array([])) if return_proba else res
    model = _clf(kind)
    if hasattr(model, "scale_pos_weight"):
        model.set_params(scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
    res = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
           "balanced_acc": float(balanced_accuracy_score(yte, pred)), "auc": float(auc)}
    return (res, yte, proba) if return_proba else res


def fit_eval_regression(Xtr, ytr, Xte, yte, kind: str = "xgb") -> dict:
    ytr, yte = np.asarray(ytr, float), np.asarray(yte, float)
    mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
    Xtr, ytr = Xtr.loc[mtr], ytr[mtr]
    Xte, yte = Xte.loc[mte], yte[mte]
    if len(Xtr) == 0 or len(Xte) == 0:
        return {"n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                "mae": float("nan"), "r2": float("nan")}
    model = _reg(kind)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"n_train": int(len(ytr)), "n_test": int(len(yte)),
            "mae": float(mean_absolute_error(yte, pred)), "r2": float(r2_score(yte, pred))}


def cv_regression(X: pd.DataFrame, y: np.ndarray, kind: str = "xgb",
                  n_splits: int = 5, seed: int = 0) -> dict:
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)
    X, y = X.loc[m], y[m]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, r2s = [], []
    for tr, te in kf.split(X):
        model = _reg(kind)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        maes.append(mean_absolute_error(y[te], pred))
        r2s.append(r2_score(y[te], pred))
    return {"n": int(m.sum()), "mae": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "r2": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}
