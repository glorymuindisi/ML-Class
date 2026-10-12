# Lab 1: Environment and first system measurements

## Goal
(1-2 sentences: what you built and measured)

## Method
(Python 3.11, .venv with pinned packages, Breast Cancer Wisconsin,
70/30 stratified split, seed 42, the two models, how you measured:
1 warm-up + median of 5 for training, median of 100 for inference,
joblib file size, psutil peak RSS)

## Results

| Model | Accuracy | Train time (s) | Latency (ms) | Size (KB) | Peak mem train (MB) | Peak mem inference (MB) |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.9415 | 0.5189 | 0.0759 | 1.0 | 102.6 | 102.7 |
| RandomForest | 0.9357 | 0.2327 | 3.2365 | 284.1 | 104.1 | 104.1 |

## Deployment fit
(the table above plus your short justification)

## Conclusions
1.
2.
3.

## Notes
- pyarrow pinned to 15.0.2 because mlflow 2.14.1 requires pyarrow<16.
- Two models, so models are saved as model_LogisticRegression.joblib and model_RandomForest.joblib.
- Memory is whole-process peak RSS, so it includes Python and the libraries.