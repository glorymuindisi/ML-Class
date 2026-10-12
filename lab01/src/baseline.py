import csv
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)

# Load the dataset
X, y = load_breast_cancer(return_X_y=True)

# 70/30 split, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.4f}")
    rows.append([name, f"{acc:.4f}"])

with open("lab01/results/baseline_accuracy.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "test_accuracy"])
    writer.writerows(rows)