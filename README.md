# Predicting Bridge Performance under Earthquakes with Supervised Learning
UE24CS352A – Machine Learning – Mini-Project

**Problem:** given a bridge and an earthquake, predict whether the bridge is **damaged** (binary) and, as an extension,
**how badly** (3 damage levels).

Based on the CS229 (Fall 2018) project by Ziyang Jiang & Xiao Zhang
(https://github.com/jzy95310/CS229-Fall-2018-Final-Report). This repo has two parts:

| Part | Script | What it does |
|---|---|---|
| **A. Reproduction** | `src/main.py` | The paper's experiment: Logistic Regression, QDA, KNN (K=2) with bootstrap resampling on age/magnitude/distance. |
| **B. Future work (implemented)** | `src/future_work.py` | Everything the paper listed as future work – see table below. |

## Future work from the paper → what was implemented
| Paper's future work | Implementation |
|---|---|
| More features (material, structure type, …) | Material type (steel/concrete) and bridge type (beam, truss, cable, arch) added as one-hot features; 3 feature sets compared. *Traffic (AADT) is not in the dataset* – it can be added through `data/extra_records.csv` workflow if you collect it (see below). |
| Other generative models | LDA, Gaussian Naive Bayes, a Gaussian-Mixture classifier (one GMM per class, `src/models.py`) and regularised QDA, alongside the paper's QDA. |
| Multi-class classification | Damage level: none / moderate-severe / partial-or-total collapse (`--task multiclass`). |
| More / more meaningful positive examples | SMOTE (synthetic minority oversampling, written from scratch in `src/resampling.py`) compared with the paper's bootstrap and with no resampling; uses all 249 records; `data/extra_records.csv` hook for genuinely new records. **No new real records were collected** – SMOTE creates synthetic points, which is not the same as new evidence. |

### Evaluation changes in part B (why the numbers differ from part A)
* The 249 records come from only **20 bridges**. A random split puts the same bridge in both train and test. Part B uses
  **StratifiedGroupKFold** (all records of one bridge stay together), repeated 5×, so every model is judged on bridges it has not seen.
* Resampling is applied to **training folds only**; scaling (z-score of age/magnitude/distance) is fitted on training folds only.
* Metrics: accuracy is misleading at 1:10 imbalance, so recall, specificity, precision, F1, balanced accuracy and ROC-AUC are reported.
* A **magnitude-only** logistic regression is included as a sanity baseline.

## Project structure
```
bridge_performance_ml/
├── data/                       ds2.csv (full data), pre-split train/test CSVs, knn_optimal/, extra_records_template.csv
├── src/
│   ├── main.py                 Part A – reproduction of the paper
│   ├── future_work.py          Part B – experiments (binary + multi-class)
│   ├── data_prep.py            parsing of the full dataset, feature construction, extra-records hook
│   ├── resampling.py           bootstrap + SMOTE
│   ├── models.py               model zoo incl. GMM classifier and magnitude-only baseline
│   └── utils.py                CSV loader / class-wise accuracy (Part A)
├── outputs/                    generated plots, tables, logs
├── requirements.txt
└── README.md
```

## Setup (Python 3.8+)
```bash
cd bridge_performance_ml
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
# Part A – paper reproduction
python src/main.py                    # full (3 models, 100 bootstrap iterations/ratio)
python src/main.py --quick
python src/main.py --find-k           # choose K for KNN
python src/main.py --features x4_x5

# Part B – future work
python src/future_work.py             # binary + multi-class, 5x5-fold bridge-wise CV (~15 s)
python src/future_work.py --task binary
python src/future_work.py --task multiclass
python src/future_work.py --quick     # 2 repeats
python src/future_work.py --cv stratified   # ordinary CV (not grouped by bridge) for comparison
```
Outputs: `outputs/future_work/{binary,multiclass}_results.csv`, `*_comparison.png`, `*_confusion_best.png`, and console tables.

## Adding your own records (optional)
Copy `data/extra_records_template.csv` to `data/extra_records.csv` and append rows
(`bridge,material,bridge_type,age,magnitude,distance,damaged,damage_level`; material ∈ Steel/Concrete;
bridge_type ∈ Beam/Girder, Truss, Cable/Suspension, Arch; damage_level 0/1/2; damaged = 1 iff level > 0).
They are picked up automatically by `future_work.py`. Cite the source of every record you add.

## Key findings (see `outputs/future_work_results.txt`)
* Under bridge-wise CV, the paper's QDA+bootstrap setup reaches recall 0.905 but only precision ≈ 0.74 (F1 ≈ 0.81).
* Best binary F1 ≈ 0.94 (KNN k=2 + SMOTE) and ≈ 0.90 for several generative/linear models (GMM, LDA, QDA, LogReg).
  With only 21 damaged cases, one extra correct bridge-event moves recall by ~5 points – differences among top models are within noise.
* Material and bridge type did **not** improve binary detection; they gave a small gain for the 3-class task (macro-F1 0.70 → 0.74).
* Multi-class is much harder: moderate vs collapse are confused (only 11 and 10 examples).
* Magnitude alone already catches every damaged case (all damaged events are M ≥ 6.6) but with ~10% false alarms; the other features mainly reduce those false alarms.
* Caveat: undamaged examples are "other earthquakes near the same bridge", so the dataset has selection bias (damaging events are large). Results show separation in this dataset, not general seismic-safety predictions.
