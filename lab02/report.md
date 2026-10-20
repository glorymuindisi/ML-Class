# Lab 1: Environment and first system measurements

## Goal
During lab 2 I have created a neural network that clasifies handwritten numbers. It takes an image and passes it through hidden layer and predicts the number. It makes a gues measures the error adjusts its weights

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
S

## Conclusions
1. the model reached 96.1%  accuracy which is pretty high 
2.model is cheap to run prediction only needs forward pass
3. it should not be used on other datasets without retraining

## Notes
- pyarrow pinned to 15.0.2 because mlflow 2.14.1 requires pyarrow<16.
- Two models, so models are saved as model_LogisticRegression.joblib and model_RandomForest.joblib.
- Memory is whole-process peak RSS, so it includes Python and the libraries.
