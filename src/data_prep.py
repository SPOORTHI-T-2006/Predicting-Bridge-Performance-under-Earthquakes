"""Data preparation for the future-work experiments.

Reads the FULL raw dataset (data/ds2.csv) - not just the pre-split train/test files - and exposes
  * material type  (x1)  : Steel / Concrete
  * bridge type    (x3)  : Beam/Girder, Truss, Cable/Suspension, Arch
  * damage level   (y2)  : 0 = no damage, 1 = moderate/severe, 2 = partial/overall collapse
and optionally appends your own extra records from data/extra_records.csv.
"""
import csv
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

MATERIAL = {"I": "Steel", "II": "Concrete"}
BRIDGE_TYPE = {"I": "Beam/Girder", "II": "Truss", "III": "Cable/Suspension", "IV": "Arch"}
LEVEL = {"I": 0, "II": 1, "III": 2}
LEVEL_NAMES = ["No damage", "Moderate/Severe", "Partial/Overall collapse"]

# fixed category order -> fixed one-hot columns (first category is the dropped baseline)
MATERIAL_CATS = ["Concrete", "Steel"]
TYPE_CATS = ["Beam/Girder", "Truss", "Cable/Suspension", "Arch"]
COLUMNS = ["bridge", "material", "bridge_type", "age", "magnitude", "distance", "damaged", "damage_level"]

# feature sets compared in the experiments: name -> extra categorical features added to the 3 numeric ones
FEATURE_SETS = {
    "base: age+mag+dist": [],
    "+ material": ["material"],
    "+ material + bridge type": ["material", "bridge_type"],
}


def load_original():
    """Parse data/ds2.csv (contains a multi-line notes cell, so use the csv module)."""
    rows = []
    with open(os.path.join(DATA, "ds2.csv"), newline="", encoding="latin-1") as f:
        for r in csv.reader(f):
            if len(r) < 9:
                continue
            try:
                y1 = float(r[7])
                age, mag, dist = float(r[3]), float(r[5]), float(r[6])
            except ValueError:
                continue  # header row, notes row, blank row
            if y1 not in (0.0, 1.0):
                continue
            code = lambda s: s.strip()
            try:
                rows.append([
                    r[0].split(",")[0].strip(), MATERIAL[code(r[2])], BRIDGE_TYPE[code(r[4])],
                    age, mag, dist, int(y1), LEVEL[code(r[8])],
                ])
            except KeyError:
                continue
    return pd.DataFrame(rows, columns=COLUMNS)


def load_extra():
    """Optional user-supplied records (same columns as COLUMNS). See data/extra_records_template.csv."""
    path = os.path.join(DATA, "extra_records.csv")
    if not os.path.exists(path):
        return None
    ex = pd.read_csv(path)
    missing = set(COLUMNS) - set(ex.columns)
    if missing:
        raise ValueError(f"extra_records.csv is missing columns: {sorted(missing)}")
    ex = ex[COLUMNS].dropna()
    if not ex.material.isin(MATERIAL_CATS).all() or not ex.bridge_type.isin(TYPE_CATS).all():
        raise ValueError(f"material must be in {MATERIAL_CATS}; bridge_type in {TYPE_CATS}")
    if not ((ex.damaged == 1) == (ex.damage_level > 0)).all():
        raise ValueError("damaged must be 1 exactly when damage_level > 0")
    return ex


def load_dataset(verbose=True):
    df = load_original()
    n_orig = len(df)
    ex = load_extra()
    if ex is not None and len(ex):
        df = pd.concat([df, ex], ignore_index=True)
    df["damaged"] = df["damaged"].astype(int)
    df["damage_level"] = df["damage_level"].astype(int)
    if verbose:
        print(f"Dataset: {len(df)} records from {df.bridge.nunique()} bridges "
              f"({n_orig} original + {len(df) - n_orig} extra)")
        print(f"  binary    : undamaged={int((df.damaged == 0).sum())}, damaged={int(df.damaged.sum())}")
        lv = df.damage_level.value_counts().sort_index()
        print("  3-class   : " + ", ".join(f"{LEVEL_NAMES[i]}={int(lv.get(i, 0))}" for i in range(3)))
    return df


def build_features(df, extras):
    """Return X (numpy), feature names, and indices of the categorical (0/1) columns.

    The first 3 columns are always the numeric features [age, magnitude, distance].
    """
    parts = [df[["age", "magnitude", "distance"]].astype(float)]
    if "material" in extras:
        m = pd.Categorical(df.material, categories=MATERIAL_CATS)
        parts.append(pd.get_dummies(m, prefix="material", drop_first=True).astype(float))
    if "bridge_type" in extras:
        t = pd.Categorical(df.bridge_type, categories=TYPE_CATS)
        parts.append(pd.get_dummies(t, prefix="type", drop_first=True).astype(float))
    X = pd.concat(parts, axis=1)
    cat_idx = list(range(3, X.shape[1]))
    return X.values, list(X.columns), cat_idx
