import csv
import os
import random
import statistics
import threading
import time
import warnings

import joblib
import numpy as np
import psutil
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)

random.seed(42)
np.random.seed(42)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

proc = psutil.Process()


def rss_mb():
    return proc.memory_info().rss / 1024**2


def peak_mb(func):
    """Peak RSS (MB) of this process while func runs, sampled every 5 ms."""
    peak = [rss_mb()]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], rss_mb())
            time.sleep(0.005)

    t = threading.Thread(target=sampler)
    t.start()
    try:
        func()
    finally:
        stop.set()
        t.join()
    return max(peak[0], rss_mb())


os.makedirs("lab01/results", exist_ok=True)
rows = []

for name, base_model in models.items():
    # Training time: 1 warm-up, then median of 5 runs
    clone(base_model).fit(X_train, y_train)
    train_times = []
    for _ in range(5):
        m = clone(base_model)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
    train_median_s = statistics.median(train_times)

    model = clone(base_model).fit(X_train, y_train)

    # Single-sample inference: 1 warm-up, then 100 repeats, median in ms
    sample = X_test[:1]
    model.predict(sample)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(sample)
        latencies.append((time.perf_counter() - t0) * 1000)
    latency_median_ms = statistics.median(latencies)

    # Model size
    path = f"lab01/results/model_{name}.joblib"
    joblib.dump(model, path)
    size_bytes = os.path.getsize(path)
    size_kb = size_bytes / 1024

    # Peak memory during training and inference
    baseline = rss_mb()
    train_peak = peak_mb(lambda: clone(base_model).fit(X_train, y_train))
    infer_peak = peak_mb(lambda: [model.predict(sample) for _ in range(100)])

    print(f"\n{name}")
    print(f"  training time (median of 5): {train_median_s:.4f} s")
    print(f"  inference latency (median of 100): {latency_median_ms:.4f} ms")
    print(f"  model size: {size_bytes} bytes = {size_kb:.1f} KB")
    print(f"  peak memory, training: {train_peak:.1f} MB")
    print(f"  peak memory, inference: {infer_peak:.1f} MB")
    print(f"  process memory before measuring: {baseline:.1f} MB")

    rows.append([
        name,
        f"{train_median_s:.4f}",
        f"{latency_median_ms:.4f}",
        size_bytes,
        f"{size_kb:.1f}",
        f"{train_peak:.1f}",
        f"{infer_peak:.1f}",
    ])

with open("lab01/results/system_cost.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "train_time_median_s", "inference_latency_median_ms",
        "model_size_bytes", "model_size_kb",
        "peak_memory_training_mb", "peak_memory_inference_mb",
    ])
    writer.writerows(rows)