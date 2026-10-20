import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(0)
os.makedirs("results", exist_ok=True)

# 1. Data
X, y = load_digits(return_X_y=True)
X = X / 16
Y = np.eye(10)[y]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(
    X, Y, y, test_size=0.3, stratify=y, random_state=42)

# 2. Parameters (std 0.1, i.e. variance 0.01)
def init(W):
    p = {}
    for i, (a, b) in enumerate([(64, W), (W, W), (W, 10)], 1):
        p[f"W{i}"] = np.random.randn(a, b) * 0.1
        p[f"b{i}"] = np.zeros(b)
    return p

# 3. Forward
def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(0, z1)
    z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(0, z2)
    z3 = a2 @ p["W3"] + p["b3"]
    e = np.exp(z3 - z3.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True), (X, z1, a1, z2, a2)

# 4. Loss
def loss_fn(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs), axis=1))

# 5. Backward
def backward(p, cache, probs, Y):
    X, z1, a1, z2, a2 = cache
    dz3 = (probs - Y) / len(X)
    dz2 = (dz3 @ p["W3"].T) * (z2 > 0)
    dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
    return {"W3": a2.T @ dz3, "b3": dz3.sum(0),
            "W2": a1.T @ dz2, "b2": dz2.sum(0),
            "W1": X.T @ dz1,  "b1": dz1.sum(0)}

# Tasks 3 and 4
p = init(16)
probs, _ = forward(p, Xtr[:32])
print(probs.shape, np.abs(probs.sum(1) - 1).max(), loss_fn(forward(p, Xtr)[0], Ytr))

# Task 5: gradient check on 10 random weights
Xb, Yb = Xtr[:32], Ytr[:32]
probs, cache = forward(p, Xb)
g = backward(p, cache, probs, Yb)
eps, passed = 1e-5, 0
for _ in range(10):
    k = np.random.choice(["W1", "W2", "W3"])
    idx = tuple(np.random.randint(s) for s in p[k].shape)
    old = p[k][idx]
    p[k][idx] = old + eps; lp = loss_fn(forward(p, Xb)[0], Yb)
    p[k][idx] = old - eps; lm = loss_fn(forward(p, Xb)[0], Yb)
    p[k][idx] = old
    num, ana = (lp - lm) / (2 * eps), g[k][idx]
    rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
    passed += rel < 1e-6
    print(k, idx, ana, num, rel)
print("gradient check passed:", passed, "/ 10")

# Task 6: training
def train(W, lr):
    p = init(W)
    losses = []
    for ep in range(50):
        perm = np.random.permutation(len(Xtr))
        tot = 0
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            probs, cache = forward(p, Xtr[b])
            tot += loss_fn(probs, Ytr[b]) * len(b)
            g = backward(p, cache, probs, Ytr[b])
            for k in p:
                p[k] -= lr * g[k]
        losses.append(tot / len(Xtr))
        print(f"W={W} lr={lr} epoch {ep+1} loss {losses[-1]:.4f}")
    acc = np.mean(forward(p, Xte)[0].argmax(1) == yte)
    print(f"W={W} lr={lr} final test accuracy {acc:.4f}")
    return losses

for W in [16, 64]:
    for lr in [0.001, 0.01, 0.1]:
        plt.plot(train(W, lr), label=f"W={W}, lr={lr}")
plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.legend()
plt.savefig("results/loss_curves.png", dpi=150)

# Task 7: timing, median of 100 runs (W=64, batch of 32)
p = init(64)
tf, tb = [], []
for _ in range(100):
    t0 = time.perf_counter(); probs, cache = forward(p, Xb)
    t1 = time.perf_counter(); backward(p, cache, probs, Yb)
    t2 = time.perf_counter()
    tf.append(t1 - t0); tb.append(t2 - t1)
print(f"forward {np.median(tf)*1e6:.1f} us, backward {np.median(tb)*1e6:.1f} us")