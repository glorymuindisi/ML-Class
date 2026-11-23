# Lab 4 Report: Data Pipeline and Quality (UCI Adult)

#Goal
Build a reproducible, leakage-free data pipeline for the UCI Adult dataset and use it to
predict whether a person earns more than $50K a year. Success is measured with accuracy
and F1 (positive class `>50K`) on a held-out test set. The pipeline must also check data
quality and be rebuilt with one command.



#Method
1. **Ingestion:** `fetch_openml(name='adult', version=2, as_frame=True)`, read without
   changes. Logged 48842 rows and 15 columns.
2. **Validation:** schema (types), ranges (age 17-90, hours 1-99, capital-gain 0-99999,
   capital-loss 0-4356), and missing values (`?` counts as missing). Violations are
   reported, and no rows are dropped.
3. **Split first:** 70/30 stratified train/test split (seed 42) before any cleaning.
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
| Logistic regression, without derived features | 0.8540 | 0.6601 |
| Logistic regression, with derived features | 0.8578 | 0.6696 |
| Change | +0.0038 | +0.0095 |

**Data quality and pipeline checks**

| Check | Result |
|---|---|
| Raw size | 48842 rows x 15 columns |
| Train / test rows | 34165 / 14653 (after dropping 24 exact duplicates from train) |
| Class balance | 23.9% of rows are `>50K` |
| Type and range violations | 0 |
| Missing values (workclass / occupation / native-country) | 2799 / 2809 / 857 |
| Duplicate feature rows with conflicting labels | 5 groups (10 rows) |
| Impossible values in the 50-row sample (automated check) | 0 |
| Processed data SHA256 | `134c1a721928df219d748df4286646d54173b22bf933a9ee3f824231aede1498` |

## Conclusions
1.The derived features improved the baseline slightly. Adding capital_net and hours_per_age raised accuracy from 0.8540 to 0.8578 (+0.0038) and F1 on the >50K class from 0.6601 to 0.6696 (+0.0095). The gain is small and comes from a single split, so it should be confirmed with repeated seeds or cross-validation before calling it a firm improvement.
2. Accuracy alone is misleading on this data. The best baseline reaches 85.8% accuracy, only 9.7 points above that, while its F1 on the >50K class is 0.670. This is why both metrics are reported.
3. Data quality and pipeline design make the scores trustworthy. Validation found 0 type or range violations. The 70/30 stratified split was done before any cleaning, and the cleaner was fitted on train data only, so no test information leaked into training. Duplicates were dropped from train only, and the processed data hash and python run_pipeline.py make the results reproducible
4. Stratified means the split keeps the same class proportions in both train and test sets. 
eg if  23.9% of people earn >50K. A random split could by chance give the test
set 22% or 26%. A stratified split forces both train and test to have about 23.9% each, 
so the two sets stay comparable and your F1 score on the rare class is more reliable. 
less stable training and unfair evaluation
we use it when we split data for a classification problem


