"""Predicting bridge performance under earthquakes with supervised learning.

Models   : Logistic Regression, Quadratic Discriminant Analysis (QDA), K-Nearest Neighbours
Imbalance: bootstrap re-sampling (with replacement) of the positive (damaged) class
Features : x2 = bridge age (yr), x4 = earthquake magnitude, x5 = distance to epicentre (miles)

Usage (from project root):
    python src/main.py                       # full experiment, all 3 models, age+magnitude+distance
    python src/main.py --features x4_x5      # magnitude + distance only
    python src/main.py --quick               # 10 bootstrap iterations (fast demo)
    python src/main.py --find-k              # cross-validate K for KNN and plot K vs accuracy
    python src/main.py --models qda knn      # run only selected models
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # save figures to files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

from utils import class_accuracy, load_csv

warnings.filterwarnings("ignore")  # QDA warns about collinear variables on tiny classes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")

RATIOS = np.linspace(0.2, 1.0, 5)  # positive : negative ratio after bootstrap
PRETTY = {"logreg": "Logistic Regression", "qda": "QDA", "knn": "KNN"}


def make_model(name, k):
    if name == "logreg":
        return LogisticRegression(solver="liblinear")
    if name == "qda":
        return QuadraticDiscriminantAnalysis()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=k)
    raise ValueError(name)


def bootstrap_train_set(X, y, ratio, rng):
    """Up-sample positives with replacement so that #pos = ratio * #neg, then merge with all negatives."""
    pos, neg = X[y == 1], X[y == 0]
    n_samples = int(round(ratio * len(neg)))
    pos_up = resample(pos, replace=True, n_samples=n_samples,
                      random_state=int(rng.integers(1_000_000_000)))
    Xb = np.vstack([pos_up, neg])
    yb = np.concatenate([np.ones(len(pos_up)), np.zeros(len(neg))])
    return Xb, yb


def run_model(name, Xtr, ytr, Xte, yte, k, iters, rng, save_dir):
    """For each ratio run `iters` bootstrap trials; plot train/test accuracy per trial."""
    os.makedirs(save_dir, exist_ok=True)
    train_acc = np.zeros((len(RATIOS), iters))
    test_acc = np.zeros((len(RATIOS), iters))
    pos_acc, neg_acc = [], []

    for r, ratio in enumerate(RATIOS):
        for i in range(iters):
            Xb, yb = bootstrap_train_set(Xtr, ytr, ratio, rng)
            model = make_model(name, k).fit(Xb, yb)
            train_acc[r, i] = accuracy_score(yb, model.predict(Xb))
            pred = model.predict(Xte)
            test_acc[r, i] = accuracy_score(yte, pred)
            pos_acc.append(class_accuracy(yte, pred, 1))
            neg_acc.append(class_accuracy(yte, pred, 0))

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, iters + 1), train_acc[r], color="blue", lw=1, label="train")
        plt.plot(range(1, iters + 1), test_acc[r], color="red", lw=1, label="test")
        plt.ylim(0.5, 1.0)
        plt.xlabel("# of iterations")
        plt.ylabel("Accuracy")
        plt.title(f"{PRETTY[name]} with #pos/#neg = {ratio:.1f}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_{ratio:.1f}.png"), dpi=120)
        plt.close()

    avg = np.column_stack([train_acc.mean(1), test_acc.mean(1)])
    np.savetxt(os.path.join(save_dir, f"{name}_accuracy_by_ratio.txt"), avg,
               header="train_acc test_acc (rows: ratios 0.2,0.4,0.6,0.8,1.0)")
    return {"train": avg[:, 0].mean(), "test": avg[:, 1].mean(),
            "pos": float(np.nanmean(pos_acc)), "neg": float(np.nanmean(neg_acc))}


def find_optimal_k(iters, rng):
    """Pick K in 1..10 by accuracy on a held-out cross-validation set (bootstrap-averaged)."""
    Xtr, ytr = load_csv(os.path.join(DATA, "knn_optimal", "ds2_knn_train.csv"))
    Xcv, ycv = load_csv(os.path.join(DATA, "knn_optimal", "ds2_knn_cv.csv"))
    ks = np.arange(1, 11)
    cv = np.zeros(len(ks))
    for j, k in enumerate(ks):
        accs = []
        for ratio in RATIOS:
            for _ in range(iters):
                Xb, yb = bootstrap_train_set(Xtr, ytr, ratio, rng)
                m = KNeighborsClassifier(n_neighbors=int(k)).fit(Xb, yb)
                accs.append(accuracy_score(ycv, m.predict(Xcv)))
        cv[j] = np.mean(accs)
    best = int(ks[np.argmax(cv)])
    plt.figure(figsize=(6, 4))
    plt.plot(ks, cv, color="blue", lw=2)
    plt.xlabel("K value")
    plt.ylabel("Cross-validation accuracy")
    plt.title("K value VS. Cross-validation accuracy")
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, "knn_optimal_k.png"), dpi=120)
    plt.close()
    print(f"\nOptimal K (by cross-validation accuracy) = {best}")
    for k, a in zip(ks, cv):
        print(f"  K={k:2d}  cv-accuracy={a:.4f}")
    return best


def plot_data(features):
    X, y = load_csv(os.path.join(DATA, f"ds2_{features}_train.csv"))
    mag, dist = X[:, -2], X[:, -1]
    plt.figure(figsize=(6, 4))
    plt.scatter(mag[y == 0], dist[y == 0], c="blue", s=14, label="undamaged (0)")
    plt.scatter(mag[y == 1], dist[y == 1], c="red", s=22, label="damaged (1)")
    plt.xlabel("Earthquake magnitude")
    plt.ylabel("Distance to epicentre (miles)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, "data_scatter.png"), dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=["x2_x4_x5", "x4_x5"], default="x2_x4_x5",
                    help="x2_x4_x5 = age+magnitude+distance (default); x4_x5 = magnitude+distance")
    ap.add_argument("--models", nargs="+", choices=["logreg", "qda", "knn"], default=["logreg", "qda", "knn"])
    ap.add_argument("--k", type=int, default=2, help="K for KNN (paper's CV result: 2)")
    ap.add_argument("--iters", type=int, default=100, help="bootstrap iterations per ratio")
    ap.add_argument("--quick", action="store_true", help="shortcut for --iters 10")
    ap.add_argument("--find-k", action="store_true", help="cross-validate K in 1..10 and exit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    iters = 10 if args.quick else args.iters
    rng = np.random.default_rng(args.seed)

    if args.find_k:
        find_optimal_k(max(iters // 5, 5), rng)
        print(f"Plot saved to {os.path.join(OUT, 'knn_optimal_k.png')}")
        return

    Xtr, ytr = load_csv(os.path.join(DATA, f"ds2_{args.features}_train.csv"))
    Xte, yte = load_csv(os.path.join(DATA, f"ds2_{args.features}_test.csv"))
    print(f"Features: {args.features} | train={len(ytr)} (pos={int(ytr.sum())}) | "
          f"test={len(yte)} (pos={int(yte.sum())}) | iters/ratio={iters} | K={args.k}")
    plot_data(args.features)

    rows = []
    for name in args.models:
        print(f"Running {PRETTY[name]} ...")
        res = run_model(name, Xtr, ytr, Xte, yte, args.k, iters, rng,
                        os.path.join(OUT, args.features, name))
        rows.append((PRETTY[name], res))

    print(f"\n=== Results (avg over 5 pos:neg ratios x {iters} bootstrap iterations) ===")
    print(f"{'Model':<22}{'Train':>8}{'Test':>8}{'Test(damaged)':>16}{'Test(undamaged)':>18}")
    lines = ["model,train_acc,test_acc,test_acc_positive,test_acc_negative"]
    for n, r in rows:
        print(f"{n:<22}{r['train']*100:>7.1f}%{r['test']*100:>7.1f}%{r['pos']*100:>15.1f}%{r['neg']*100:>17.1f}%")
        lines.append(f"{n},{r['train']:.4f},{r['test']:.4f},{r['pos']:.4f},{r['neg']:.4f}")
    with open(os.path.join(OUT, args.features, "summary.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nPlots and summary.csv saved in {os.path.join(OUT, args.features)}")


if __name__ == "__main__":
    main()
