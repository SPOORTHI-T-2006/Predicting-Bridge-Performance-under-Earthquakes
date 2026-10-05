# Predicting Bridge Performance under Earthquakes with Supervised Learning

**Course:** UE24CS352A – Machine Learning – Mini-Project


**Problem:** Given a bridge (age, material, type) and an earthquake (magnitude, distance from the epicentre),
predict whether the bridge will be **damaged** (binary classification) and, as an extension,
**how badly** (3-class: none / moderate-severe / collapse).

## Attribution and our contributions
**Source:** The topic, dataset and baseline experiment (Logistic Regression, QDA and KNN with bootstrap
resampling) come from Jiang & Zhang, *Predicting Bridge Performance under Earthquakes with Supervised Learning*,
CS229 Fall 2018 – https://github.com/jzy95310/CS229-Fall-2018-Final-Report

**Our work:**
- **Part A – Baseline:** re-implemented the paper's experiment as one command-line pipeline (`src/main.py`).
- **Part B – Extension:** implemented the paper's own future-work list (`src/future_work.py`): extra features
  (material, bridge type), other generative models (LDA, Gaussian Naive Bayes, Gaussian-Mixture classifier,
  regularised QDA), SMOTE oversampling, and multi-class damage-level prediction.
- Evaluation: bridge-wise repeated cross-validation (no leakage between records of the same bridge),
  extra metrics (recall, precision, F1, ROC-AUC) and a magnitude-only sanity baseline.
- Analysis of results and limitations.

## Project structure
```
├── data/        ds2.csv (full dataset), pre-split train/test CSVs, knn_optimal/, extra_records_template.csv
├── src/
│   ├── main.py          Part A – paper reproduction
│   ├── future_work.py   Part B – experiments (binary + multi-class)
│   ├── data_prep.py     parses the full dataset, builds features, optional extra records
│   ├── resampling.py    bootstrap and SMOTE
│   ├── models.py        model collection (incl. GMM classifier and magnitude-only baseline)
│   └── utils.py         CSV loader and class-wise accuracy (Part A)
├── outputs/     generated plots, tables and result logs
├── requirements.txt
└── README.md
```

## Setup (Python 3.8+)
```
git clone <repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```
# Part A – paper reproduction
python src/main.py                    # full: 3 models x 5 ratios x 100 bootstrap iterations
python src/main.py --quick            # fast demo (10 iterations)
python src/main.py --find-k           # choose K for KNN by cross-validation
python src/main.py --features x4_x5   # magnitude + distance only (no age)
python src/main.py --models qda knn   # selected models only

# Part B – future work (about 15-25 s)
python src/future_work.py                       # binary + multi-class
python src/future_work.py --task binary
python src/future_work.py --task multiclass
python src/future_work.py --quick               # 2 repeats instead of 5
python src/future_work.py --cv stratified       # ordinary CV instead of grouped by bridge
```
Use `python3` / `py` if `python` is not found. `--help` lists all options.
Note: `--cv stratified` overwrites the saved binary results; re-run the default command to restore them.

## Outputs
- Console tables, and `outputs/<...>/summary.csv` (Part A).
- `outputs/future_work/{binary,multiclass}_results.csv` – full results table.
- `outputs/future_work/*_comparison.png`, `*_confusion_best.png` – plots.
- `outputs/future_work_results.txt` – saved console log of the default Part B run.

## Part A results (age + magnitude + distance; matches the paper)
| Model | Train | Test | Test (damaged) | Test (undamaged) |
|---|---|---|---|---|
| Logistic Regression | ~95% | ~74% | ~84% | ~73% |
| QDA | ~98% | ~95% | ~99% | ~94% |
| KNN (K=2) | ~99% | ~81% | ~30% | ~84% |

The test set has only 50 rows (3 damaged), so these numbers are noisy.

## Part B – why the evaluation is different
The 249 records come from only **20 bridges**. A random split could put the same bridge in both train and test,
so Part B keeps all records of a bridge in one fold (StratifiedGroupKFold), repeats 5-fold cross-validation 5 times,
rebalances and scales **only training folds**, and reports recall/precision/F1/AUC instead of accuracy alone.

## Part B key findings (bridge-wise cross-validation)
- Paper-style QDA + bootstrap: recall 0.905, precision 0.737, F1 0.812.
- Best binary F1 ≈ 0.94 (KNN k=2 + SMOTE); GMM, LDA, QDA and Logistic Regression reach ≈ 0.89–0.91.
  With only 21 damaged cases, one extra correct event shifts recall by ~5 points, so top models are effectively tied.
- Material and bridge type did not improve binary detection; small gain for 3-class (macro-F1 0.694 → 0.742).
- Multi-class is hard: moderate vs collapse are confused (only 11 and 10 examples).
- Magnitude alone catches every damaged case (all damaged events have M ≥ 6.6) with ~10% false alarms.

## Limitations
- Very small data: 21 damaged examples from 20 bridges. No new real records were collected; SMOTE creates synthetic points.
- Selection bias: undamaged examples are other earthquakes near the same bridges, so "large earthquake" is an easy shortcut.
  Results describe separation in this dataset, not general seismic safety.
- No traffic (AADT) data in the dataset. QDA regularisation (0.2), SMOTE neighbours (k=5) and resampling ratio (1.0) were not tuned.

## Adding your own records (optional)
Copy `data/extra_records_template.csv` to `data/extra_records.csv` and append rows
(`bridge,material,bridge_type,age,magnitude,distance,damaged,damage_level`;
material ∈ Steel/Concrete; bridge_type ∈ Beam/Girder, Truss, Cable/Suspension, Arch; damage_level 0/1/2;
damaged = 1 exactly when damage_level > 0). Cite the source of every record.
