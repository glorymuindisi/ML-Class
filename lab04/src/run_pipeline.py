"""Lab 4: leakage-free data pipeline for UCI Adult.  Run:  python run_pipeline.py"""
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CFG = yaml.safe_load(open("config.yaml"))
OUT = Path(CFG["paths"]["results"])
OUT.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(OUT / "pipeline.log", mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("lab4")

NUM = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation", "relationship",
       "race", "sex", "native-country"]
FEATURES = [c for c in
            ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
             "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
             "hours-per-week", "native-country"]]
TARGET = "income"
POSITIVE = ">50K"

SCHEMA = {**{c: "int" for c in NUM}, **{c: "category" for c in CAT}, TARGET: "category"}
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}


def to_py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


# ---------------------------------------------------------------- 1. ingestion
def ingest():
    ds = fetch_openml(name=CFG["dataset"]["name"], version=CFG["dataset"]["version"],
                      as_frame=True)
    raw = ds.frame  # read only: no changes are made to this object
    rows, cols = raw.shape
    log.info("INGESTION: row count = %d, column count = %d", rows, cols)
    if (rows, cols) != (48842, 15):
        log.warning("Expected (48842, 15) but got (%d, %d)", rows, cols)
    return raw


# --------------------------------------------------------------- 2. validation
def is_missing(s):
    s = s.astype(object)
    return s.isna() | s.isin(["?"])


def validate(raw):
    df = raw.rename(columns={"class": TARGET})  # only a view with the lab's column name
    report = {"rows": len(df), "columns": df.shape[1], "type_violations": [],
              "range_violations": {}, "missing": {}, "categories": {}}
    for col, kind in SCHEMA.items():
        if col not in df:
            report["type_violations"].append(f"{col}: column missing")
            continue
        s = df[col]
        if kind == "int":
            ok = pd.api.types.is_numeric_dtype(s) and bool((s.dropna() % 1 == 0).all())
        else:
            ok = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object
        if not ok:
            report["type_violations"].append(f"{col}: expected {kind}, found {s.dtype}")
    for col, (lo, hi) in RANGES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        report["range_violations"][col] = {
            "allowed": [lo, hi], "violations": int(len(bad)),
            "example_row_indices": [int(i) for i in bad.index[:5]]}
    for col in CAT + [TARGET]:
        m = is_missing(df[col])
        report["missing"][col] = int(m.sum())
        report["categories"][col] = sorted(
            str(v) for v in df[col].astype(object).dropna().unique())
    for col in NUM:
        report["missing"][col] = int(df[col].isna().sum())
    n_viol = (len(report["type_violations"])
              + sum(v["violations"] for v in report["range_violations"].values()))
    report["total_violations"] = n_viol
    log.info("VALIDATION: %d violations (rows are NOT dropped). Missing per column: %s",
             n_viol, {k: v for k, v in report["missing"].items() if v})
    json.dump(report, open(OUT / "validation_report.json", "w"), indent=2, default=to_py)
    return df


# ------------------------------------------------------------- 3. label quality
def label_audit(df):
    g = df.groupby(FEATURES, observed=True, dropna=False)[TARGET].nunique()
    conflicts = g[g > 1]
    dup_sizes = df.groupby(FEATURES, observed=True, dropna=False).size()
    rows_in_conflict = int(dup_sizes.loc[conflicts.index].sum()) if len(conflicts) else 0

    sample = df.sample(CFG["audit"]["sample_size"], random_state=CFG["seed"])
    sample.to_csv(OUT / "label_audit_sample50.csv")  # for the manual re-check

    edu_map = df.groupby("education", observed=True)["education-num"].nunique()
    inconsistent_edu = [str(k) for k in edu_map[edu_map > 1].index]
    impossible = 0
    for col, (lo, hi) in RANGES.items():
        impossible += int(((sample[col] < lo) | (sample[col] > hi)).sum())
    res = {
        "duplicate_conflict_groups": int(len(conflicts)),
        "rows_in_conflicting_groups": rows_in_conflict,
        "random_sample_size": len(sample),
        "impossible_values_in_sample": impossible,
        "education_vs_education_num_inconsistencies": inconsistent_edu,
        "note": "Sample saved to label_audit_sample50.csv. Re-check these 50 rows by hand "
                "against the data dictionary and add your findings to your report.",
    }
    log.info("LABEL AUDIT: %d conflicting duplicate groups (%d rows)",
             len(conflicts), rows_in_conflict)
    json.dump(res, open(OUT / "label_audit.json", "w"), indent=2, default=to_py)
    return res


# ----------------------------------------------------------------- 4. cleaning
class Cleaner(BaseEstimator, TransformerMixin):
    """Impute (mode / median) and clip to 1st-99th percentile. Fit on train only."""

    def __init__(self, num_cols, cat_cols, lower=0.01, upper=0.99):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.lower = lower
        self.upper = upper

    def _norm(self, X):
        X = X[self.num_cols + self.cat_cols].copy()
        for c in self.cat_cols:
            s = X[c].astype(object)
            X[c] = s.where(~s.isin(["?"]), np.nan)  # '?' means missing
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c])
        return X

    def fit(self, X, y=None):
        X = self._norm(X)
        self.medians_ = X[self.num_cols].median()
        self.modes_ = X[self.cat_cols].mode(dropna=True).iloc[0]
        self.low_ = X[self.num_cols].quantile(self.lower)
        self.high_ = X[self.num_cols].quantile(self.upper)
        return self

    def transform(self, X):
        X = self._norm(X)
        X[self.num_cols] = X[self.num_cols].fillna(self.medians_)
        X[self.cat_cols] = X[self.cat_cols].fillna(self.modes_)
        X[self.num_cols] = X[self.num_cols].clip(self.low_, self.high_, axis=1)
        return X


def add_features(X):
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


def make_model(num, cat):
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def clean_and_model(df):
    X, y = df[FEATURES], df[TARGET].astype(str)
    # split FIRST (stratified 70/30), before any cleaning
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=CFG["split"]["test_size"], stratify=y, random_state=CFG["seed"])
    log.info("SPLIT: train=%d, test=%d", len(Xtr), len(Xte))

    # drop exact duplicates from TRAIN only (features + label); test untouched
    train_full = Xtr.assign(**{TARGET: ytr})
    n_before = len(train_full)
    train_full = train_full.drop_duplicates()
    log.info("DUPLICATES: dropped %d exact duplicate rows from train; test untouched",
             n_before - len(train_full))
    Xtr, ytr = train_full[FEATURES], train_full[TARGET]

    cleaner = Cleaner(NUM, CAT, CFG["cleaning"]["clip_lower"], CFG["cleaning"]["clip_upper"])
    cleaner.fit(Xtr)  # statistics come from train only
    joblib.dump(cleaner, OUT / "preprocessor.joblib")
    Ctr, Cte = cleaner.transform(Xtr), cleaner.transform(Xte)

    # baseline logistic regression, with and without derived features
    scores = {}
    base_cols = (NUM, CAT)
    der_num = NUM + ["capital_net", "hours_per_age"]
    for name, (tr, te, num) in {
        "without_derived_features": (Ctr, Cte, NUM),
        "with_derived_features": (add_features(Ctr), add_features(Cte), der_num),
    }.items():
        model = make_model(num, CAT).fit(tr, ytr)
        pred = model.predict(te)
        scores[name] = {"accuracy": accuracy_score(yte, pred),
                        "f1_>50K": f1_score(yte, pred, pos_label=POSITIVE)}
        log.info("BASELINE %s: %s", name, scores[name])
    json.dump(scores, open(OUT / "baseline_metrics.json", "w"), indent=2, default=to_py)

    # processed dataset (cleaned + derived features) for hashing
    tr_out = add_features(Ctr).assign(**{TARGET: ytr.values, "split": "train"})
    te_out = add_features(Cte).assign(**{TARGET: yte.values, "split": "test"})
    processed = pd.concat([tr_out, te_out], ignore_index=True)
    path = OUT / "processed_adult.csv"
    processed.to_csv(path, index=False, float_format="%.6f")
    return processed, path, scores, n_before - len(train_full), (len(Xtr), len(Xte))


# ------------------------------------------------- 5. hash, manifest, data card
def finish(raw, path, scores, dropped, sizes, audit, validation_total):
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    log.info("PROCESSED DATA SHA256: %s", sha)
    manifest = {"config": CFG, "processed_file": str(path), "sha256": sha,
                "train_rows_after_dedupe": sizes[0], "test_rows": sizes[1]}
    yaml.safe_dump(manifest, open(OUT / "run_manifest.yaml", "w"), sort_keys=False)

    pos_rate = (raw["class"] == POSITIVE).mean()
    card = f"""# Data Card: UCI Adult (OpenML version 2)

## Source
UCI Adult (Census Income) dataset, created by Ronny Kohavi and Barry Becker from the
1994 US Census database. Loaded through OpenML with
`fetch_openml(name='adult', version=2, as_frame=True)`.

## License
The UCI Machine Learning Repository lists Adult under CC BY 4.0. Check the dataset page
before redistributing.

## Size
- Raw: {raw.shape[0]} rows x {raw.shape[1]} columns (14 features + target).
- Processed: {sizes[0]} train rows (after dropping {dropped} exact duplicates) and {sizes[1]} test rows.
- Target: income >50K is the positive class ({pos_rate:.1%} of rows).

## Processing summary
70/30 stratified split before cleaning; mode/median imputation, 1st/99th percentile
clipping and duplicate removal use the training set only; fitted cleaner saved at
`results/preprocessor.joblib`. Derived features: `capital_net`, `hours_per_age`.
Processed file SHA256: `{sha}`.

## Quality checks
- Schema violations (types and ranges): {validation_total}
- Duplicate feature rows with conflicting labels: {audit['duplicate_conflict_groups']} groups ({audit['rows_in_conflicting_groups']} rows)
- Random sample of 50 rows saved for manual review; automated impossible values found: {audit['impossible_values_in_sample']}

## Baseline results (logistic regression, test set)
- Without derived features: accuracy {scores['without_derived_features']['accuracy']:.4f}, F1 {scores['without_derived_features']['f1_>50K']:.4f}
- With derived features: accuracy {scores['with_derived_features']['accuracy']:.4f}, F1 {scores['with_derived_features']['f1_>50K']:.4f}

## Known biases
- Data comes from the 1994 US Census, so it does not reflect today's incomes or economy.
- Strong imbalance by sex and race (mostly male and White), and the labels show a
  large income gap between groups. A model can learn and repeat these patterns.
- Income is cut at a single $50K threshold, and that value is not adjusted for inflation.
- `capital-gain` is capped at 99999, a placeholder-like top value.
- `fnlwgt` is a census sampling weight, not a personal attribute.
- Missing values (workclass, occupation, native-country) are not random.

## Privacy concerns
The data is public and has no names, but combinations of age, sex, race, country,
occupation and education can act as quasi-identifiers and may re-identify people when
joined with other data. Do not try to link it to other datasets.

## Reproducibility
Run `python run_pipeline.py` (seed {CFG['seed']}, settings in `config.yaml`).
"""
    (OUT / "data_card.md").write_text(card)
    write_report(raw, scores, dropped, sizes, audit, sha)


# ----------------------------------------------------------------- 6. report.md
def write_report(raw, scores, dropped, sizes, audit, sha):
    """Write results/report.md: goal, method, results tables, 3 conclusions.
    Every number is read from this run, so the report always matches the data."""
    val = json.load(open(OUT / "validation_report.json"))
    miss = val["missing"]
    pos_rate = (raw["class"] == POSITIVE).mean()
    majority = 1 - pos_rate

    wo, wi = scores["without_derived_features"], scores["with_derived_features"]
    wo_acc, wo_f1 = wo["accuracy"], wo["f1_>50K"]
    wi_acc, wi_f1 = wi["accuracy"], wi["f1_>50K"]
    d_acc, d_f1 = wi_acc - wo_acc, wi_f1 - wo_f1
    best = max(scores.values(), key=lambda s: s["f1_>50K"])

    # conclusion 1: did the derived features help?
    if d_f1 > 0.002:
        c1 = (f"The derived features improved the baseline: F1 rose from {wo_f1:.4f} to "
              f"{wi_f1:.4f} ({d_f1:+.4f}) and accuracy changed by {d_acc:+.4f}.")
    elif d_f1 < -0.002:
        c1 = (f"The derived features did not help: F1 fell from {wo_f1:.4f} to "
              f"{wi_f1:.4f} ({d_f1:+.4f}) and accuracy changed by {d_acc:+.4f}.")
    else:
        c1 = (f"The derived features made almost no difference: F1 changed by {d_f1:+.4f} "
              f"and accuracy by {d_acc:+.4f}.")
    c1 += (" `capital_net` is a linear mix of two existing columns, so logistic regression "
           "gets little new information from it; `hours_per_age` is the only genuinely "
           "new signal.")

    # conclusion 2: accuracy alone is misleading
    c2 = (f"Accuracy alone is misleading on this data. About {pos_rate:.1%} of people earn "
          f">50K, so always predicting `<=50K` already scores about {majority:.1%} accuracy. "
          f"The best baseline reaches {best['accuracy']:.1%} accuracy, only "
          f"{(best['accuracy'] - majority) * 100:.1f} points above that, while its F1 on the "
          f">50K class is {best['f1_>50K']:.3f}. This is why both metrics are reported.")

    # conclusion 3: data quality and leakage
    c3 = (f"Data quality and pipeline design make the scores trustworthy. Validation found "
          f"{val['total_violations']} type or range violations. Missing values were mainly in "
          f"workclass ({miss['workclass']}), occupation ({miss['occupation']}) and "
          f"native-country ({miss['native-country']}), and they were imputed with statistics "
          f"from the training set only. ")
    if audit["duplicate_conflict_groups"] > 0:
        c3 += (f"The audit found {audit['duplicate_conflict_groups']} groups of identical "
               f"feature rows with different income labels "
               f"({audit['rows_in_conflicting_groups']} rows), which is label noise that "
               f"limits the accuracy any model can reach. ")
    else:
        c3 += "The audit found no identical feature rows with conflicting labels. "
    c3 += ("Because the split was done first and everything was fitted on the training set "
           "only, the test scores are not inflated by leakage.")

    report = f"""# Lab 4 Report: Data Pipeline and Quality (UCI Adult)

## Goal
Build a reproducible, leakage-free data pipeline for the UCI Adult dataset and use it to
predict whether a person earns more than $50K a year. Success is measured with accuracy
and F1 (positive class `>50K`) on a held-out test set. The pipeline must also check data
quality and be rebuilt with one command.

## Method
1. **Ingestion:** `fetch_openml(name='adult', version=2, as_frame=True)`, read without
   changes. Logged {raw.shape[0]} rows and {raw.shape[1]} columns.
2. **Validation:** schema (types), ranges (age 17-90, hours 1-99, capital-gain 0-99999,
   capital-loss 0-4356), and missing values (`?` counts as missing). Violations are
   reported, and no rows are dropped.
3. **Split first:** {int((1 - CFG['split']['test_size']) * 100)}/{int(CFG['split']['test_size'] * 100)} stratified train/test split (seed {CFG['seed']}) before any cleaning.
4. **Cleaning (train statistics only):** median imputation for numeric columns, mode for
   categorical columns, clipping to the 1st/99th percentile, and exact duplicates dropped
   from train only. The test set is untouched. The fitted cleaner is saved to
   `results/preprocessor.joblib`.
5. **Features:** `capital_net = capital-gain - capital-loss` and
   `hours_per_age = hours-per-week / (age - 16)`.
6. **Model:** logistic regression (standardised numeric columns, one-hot categorical
   columns), trained with and without the derived features.
7. **Label quality:** duplicate-conflict audit plus a random sample of 50 rows saved for a
   manual check against the data dictionary.
8. **Reproducibility:** SHA256 hash of the processed dataset, `config.yaml`, and one command:
   `python run_pipeline.py`.

## Results

**Baseline model (test set)**

| Model | Accuracy | F1 (>50K) |
|---|---|---|
| Logistic regression, without derived features | {wo_acc:.4f} | {wo_f1:.4f} |
| Logistic regression, with derived features | {wi_acc:.4f} | {wi_f1:.4f} |
| Change | {d_acc:+.4f} | {d_f1:+.4f} |

**Data quality and pipeline checks**

| Check | Result |
|---|---|
| Raw size | {raw.shape[0]} rows x {raw.shape[1]} columns |
| Train / test rows | {sizes[0]} / {sizes[1]} (after dropping {dropped} exact duplicates from train) |
| Class balance | {pos_rate:.1%} of rows are `>50K` |
| Type and range violations | {val['total_violations']} |
| Missing values (workclass / occupation / native-country) | {miss['workclass']} / {miss['occupation']} / {miss['native-country']} |
| Duplicate feature rows with conflicting labels | {audit['duplicate_conflict_groups']} groups ({audit['rows_in_conflicting_groups']} rows) |
| Impossible values in the 50-row sample (automated check) | {audit['impossible_values_in_sample']} |
| Processed data SHA256 | `{sha}` |

## Conclusions
1. {c1}
2. {c2}
3. {c3}
"""
    (OUT / "report.md").write_text(report)
    log.info("REPORT written to %s", OUT / "report.md")


def main():
    raw = ingest()
    df = validate(raw)
    with open(OUT / "validation_report.json") as f:
        total_viol = json.load(f)["total_violations"]
    audit = label_audit(df)
    _, path, scores, dropped, sizes = clean_and_model(df)
    finish(raw, path, scores, dropped, sizes, audit, total_viol)
    log.info("DONE. Outputs are in %s/", OUT)


if __name__ == "__main__":
    main()