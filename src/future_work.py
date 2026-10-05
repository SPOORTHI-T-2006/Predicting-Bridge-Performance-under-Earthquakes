"""Future work from the paper, implemented.

  1. More / better data   -> uses ALL 249 records, optional extra_records.csv, SMOTE synthetic oversampling,
                             evaluation split by bridge (no leakage) with repeated cross-validation
  2. More features        -> + material type, + bridge type (AADT/traffic is not in the dataset)
  3. Other generative models -> LDA, Gaussian Naive Bayes, Gaussian-Mixture classifier, regularised QDA
  4. Multi-class          -> predict damage level: none / moderate-severe / collapse

Usage (project root):
    python src/future_work.py                    # both tasks, 5 repeats of 5-fold CV
    python src/future_work.py --task binary
    python src/future_work.py --task multiclass
    python src/future_work.py --quick            # 2 repeats (fast demo)
    python src/future_work.py --cv stratified    # ordinary stratified CV instead of grouped-by-bridge
"""
import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_prep import FEATURE_SETS, LEVEL_NAMES, build_features, load_dataset
from models import GENERATIVE, get_models
from resampling import rebalance

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs", "future_work")
STRATEGIES = ["none", "bootstrap", "smote"]
RATIO = 1.0  # minority classes raised to 100% of the majority class size


# ---------------------------------------------------------------- metrics
def binary_metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "balanced_acc": (rec + spec) / 2, "recall": rec,
            "specificity": spec, "precision": prec, "f1": f1, "roc_auc": roc_auc_score(y, prob[:, 1])}


def multi_metrics(y, pred, prob):
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(y == pred)), "balanced_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
            "recall_none": rec[0], "recall_moderate": rec[1], "recall_collapse": rec[2]}


def full_proba(model, X, n_classes):
    """predict_proba with columns aligned to class ids 0..n_classes-1."""
    p = np.zeros((len(X), n_classes))
    p[:, model.classes_.astype(int)] = model.predict_proba(X)
    return p


# ---------------------------------------------------------------- cross-validation
def make_splits(X, y, groups, cv, seed, n_splits=5):
    if cv == "group":  # all records of one bridge stay in the same fold
        return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups))
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))


def run_task(task, df, cv, repeats, seed):
    binary = task == "binary"
    y = (df.damaged if binary else df.damage_level).values.astype(int)
    n_classes = 2 if binary else 3
    metric_fn = binary_metrics if binary else multi_metrics
    groups = df.bridge.values
    models = get_models()
    rows, cms = [], {}

    for fs_name, extras in FEATURE_SETS.items():
        X, _, cat_idx = build_features(df, extras)
        for r in range(repeats):
            splits = make_splits(X, y, groups, cv, seed + r)
            for strat in STRATEGIES:
                oof = {m: np.zeros(len(y), int) for m in models}
                oofp = {m: np.zeros((len(y), n_classes)) for m in models}
                for f, (tr, te) in enumerate(splits):
                    rng = np.random.default_rng(seed * 10_000 + r * 100 + f)
                    Xtr, Xte = X[tr].copy(), X[te].copy()
                    sc = StandardScaler().fit(Xtr[:, :3])  # scale numeric columns using training fold only
                    Xtr[:, :3], Xte[:, :3] = sc.transform(Xtr[:, :3]), sc.transform(Xte[:, :3])
                    Xb, yb = rebalance(Xtr, y[tr], strat, RATIO, cat_idx, rng)
                    for m, make in models.items():
                        try:
                            mdl = make().fit(Xb, yb)
                            oofp[m][te] = full_proba(mdl, Xte, n_classes)
                            oof[m][te] = mdl.classes_[np.argmax(oofp[m][te][:, mdl.classes_.astype(int)], axis=1)]
                        except Exception:  # degenerate fold: fall back to majority class
                            oof[m][te] = np.bincount(y[tr]).argmax()
                            oofp[m][te] = np.eye(n_classes)[oof[m][te]]
                for m in models:
                    met = metric_fn(y, oof[m], oofp[m])
                    rows.append({"features": fs_name, "resampling": strat, "model": m, "repeat": r, **met})
                    key = (fs_name, strat, m)
                    cms[key] = cms.get(key, 0) + confusion_matrix(y, oof[m], labels=list(range(n_classes)))
    return pd.DataFrame(rows), cms


def aggregate(res):
    g = res.drop(columns="repeat").groupby(["features", "resampling", "model"])
    mean, std = g.mean(), g.std().fillna(0)
    agg = mean.join(std, rsuffix="_std").reset_index()
    return agg


# ---------------------------------------------------------------- plots
def bar_panels(agg, metrics, title, path):
    fs_names = list(FEATURE_SETS)
    model_names = list(get_models())
    fig, axes = plt.subplots(len(metrics), len(fs_names), figsize=(5.2 * len(fs_names), 3.6 * len(metrics)),
                             sharey="row", squeeze=False)
    width = 0.27
    for i, met in enumerate(metrics):
        for j, fs in enumerate(fs_names):
            ax = axes[i][j]
            for s, strat in enumerate(STRATEGIES):
                sub = agg[(agg.features == fs) & (agg.resampling == strat)].set_index("model").loc[model_names]
                ax.bar(np.arange(len(model_names)) + (s - 1) * width, sub[met], width,
                       yerr=sub[met + "_std"], capsize=2, label=strat)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=60, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(fs, fontsize=10)
            if j == 0:
                ax.set_ylabel(met)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper right", ncol=3, title="resampling", fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cm(cm, labels, title, path):
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues")
    for a in range(len(labels)):
        for b in range(len(labels)):
            ax.text(b, a, f"{cm_n[a, b]:.2f}\n({int(cm[a, b])})", ha="center", va="center",
                    color="white" if cm_n[a, b] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- reporting
def fmt(agg, cols):
    out = agg[["features", "resampling", "model"]].copy()
    for c in cols:
        out[c] = [f"{m:.3f}±{s:.3f}" for m, s in zip(agg[c], agg[c + "_std"])]
    return out


def report_binary(agg_all, cms, cv="group"):
    cols = ["accuracy", "balanced_acc", "recall", "specificity", "precision", "f1", "roc_auc"]
    agg = agg_all[agg_all.model != "MagnitudeOnly"]  # the baseline is reported separately below
    print("\n=== BINARY (damaged vs undamaged): top 10 configurations by F1 on the damaged class ===")
    top = agg.sort_values("f1", ascending=False).head(10)
    print(fmt(top, cols).to_string(index=False))
    base = agg[(agg.features == "base: age+mag+dist") & (agg.resampling == "bootstrap") & (agg.model == "QDA")]
    how = "bridge-wise (grouped)" if cv == "group" else "ordinary stratified"
    print(f"\nPaper-style baseline (QDA, age+mag+dist, bootstrap) under the same {how} evaluation:")
    print(fmt(base, cols).to_string(index=False))
    mo = agg_all[(agg_all.model == "MagnitudeOnly") & (agg_all.features == "base: age+mag+dist")].sort_values("f1", ascending=False).iloc[0]
    print(f"\nSanity baseline - logistic regression on magnitude ALONE (best resampling = {mo.resampling}): "
          f"F1={mo.f1:.3f}, recall={mo.recall:.3f}, specificity={mo.specificity:.3f}, AUC={mo.roc_auc:.3f}")
    print("\nEffect of features (best F1 within each feature set; baseline excluded):")
    for fs in FEATURE_SETS:
        b = agg[agg.features == fs].sort_values("f1", ascending=False).iloc[0]
        print(f"  {fs:<28} {b.model:<11} {b.resampling:<9} F1={b.f1:.3f}  recall={b.recall:.3f}  specificity={b.specificity:.3f}")
    print("\nGenerative models only (best config of each):")
    for m in sorted(GENERATIVE):
        b = agg[agg.model == m].sort_values("f1", ascending=False).iloc[0]
        print(f"  {m:<11} F1={b.f1:.3f}  bal.acc={b.balanced_acc:.3f}  AUC={b.roc_auc:.3f}  ({b.features}, {b.resampling})")
    best = top.iloc[0]
    key = (best.features, best.resampling, best.model)
    plot_cm(cms[key], ["Undamaged", "Damaged"], f"Best: {best.model} | {best.features} | {best.resampling}\n(summed over repeats)",
            os.path.join(OUT, "binary_confusion_best.png"))
    bar_panels(agg_all, ["balanced_acc", "f1"], "Binary task - bridge-wise cross-validation (mean ± std over repeats)",
               os.path.join(OUT, "binary_comparison.png"))


def report_multiclass(agg_all, cms):
    cols = ["accuracy", "balanced_acc", "macro_f1", "recall_none", "recall_moderate", "recall_collapse"]
    print("\n=== MULTI-CLASS (none / moderate-severe / collapse): top 10 configurations by macro-F1 ===")
    agg = agg_all[agg_all.model != "MagnitudeOnly"]
    top = agg.sort_values("macro_f1", ascending=False).head(10)
    print(fmt(top, cols).to_string(index=False))
    print("\nEffect of features (best macro-F1 within each feature set):")
    for fs in FEATURE_SETS:
        b = agg[agg.features == fs].sort_values("macro_f1", ascending=False).iloc[0]
        print(f"  {fs:<28} {b.model:<11} {b.resampling:<9} macro-F1={b.macro_f1:.3f}")
    best = top.iloc[0]
    key = (best.features, best.resampling, best.model)
    plot_cm(cms[key], LEVEL_NAMES, f"Best: {best.model} | {best.features} | {best.resampling}\n(summed over repeats)",
            os.path.join(OUT, "multiclass_confusion_best.png"))
    bar_panels(agg_all, ["macro_f1", "balanced_acc"], "Multi-class task - bridge-wise cross-validation (mean ± std over repeats)",
               os.path.join(OUT, "multiclass_comparison.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["binary", "multiclass", "all"], default="all")
    ap.add_argument("--cv", choices=["group", "stratified"], default="group",
                    help="group = all records of a bridge in one fold (default, no leakage)")
    ap.add_argument("--repeats", type=int, default=5, help="repeats of 5-fold CV with different shuffles")
    ap.add_argument("--quick", action="store_true", help="2 repeats")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    repeats = 2 if a.quick else a.repeats
    os.makedirs(OUT, exist_ok=True)

    df = load_dataset()
    print(f"Evaluation: {repeats} x 5-fold CV, split = {a.cv}; resampling applied to training folds only.")
    tasks = ["binary", "multiclass"] if a.task == "all" else [a.task]
    for t in tasks:
        t0 = time.time()
        res, cms = run_task(t, df, a.cv, repeats, a.seed)
        agg = aggregate(res)
        agg.to_csv(os.path.join(OUT, f"{t}_results.csv"), index=False)
        if t == "binary":
            report_binary(agg, cms, a.cv)
        else:
            report_multiclass(agg, cms)
        print(f"[{t} finished in {time.time() - t0:.0f}s]")
    print(f"\nAll tables (CSV) and plots saved in {OUT}")


if __name__ == "__main__":
    main()
