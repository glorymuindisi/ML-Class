
# ============================================================
# LAB 5: FRAMEWORKS AND AUTOMATIC DIFFERENTIATION
# TASKS 1–6
# ============================================================


import math
import time
import statistics
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# ============================================================
# TASK 1 & 2: VALUE NODE AND OPERATIONS
# ============================================================

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # Power
    def __pow__(self, exponent):
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    # ReLU
    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # Additional operations for Task 5
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        if self.data <= 0:
            raise ValueError("log requires a positive input")

        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ========================================================
    # TASK 3: BACKPROPAGATION
    # ========================================================

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build(parent)

                topo.append(node)

        build(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


def value_sum(values):
    values = list(values)

    if not values:
        return Value(0.0)

    result = values[0]

    for value in values[1:]:
        result = result + value

    return result


# ============================================================
# TASK 1 & 2: TESTS
# ============================================================

a = Value(5)
b = Value(3)

print("Task 1:", a, b)

x = Value(2)
y = Value(-3)

assert (x + y).data == -1
assert (x * y).data == -6
assert (x ** 2).data == 4
assert x.relu().data == 2
assert y.relu().data == 0

print("Tasks 1 and 2 passed.")


# ============================================================
# TASK 3: TEST GRADIENT ACCUMULATION
# ============================================================

x = Value(3.0)

f = x * x + x

f.backward()

print("Output:", f.data)
print("Gradient:", x.grad)

assert f.data == 12.0
assert x.grad == 7.0

print("Task 3 passed.")


# ============================================================
# TASK 4: GRADIENT VERIFICATION
# ============================================================

def custom_expression(a_values, b_values, c_values):

    aa = [Value(v) for v in a_values]
    bb = [Value(v) for v in b_values]
    cc = [Value(v) for v in c_values]

    loss = value_sum([
        ((a * b + c) ** 2).relu()
        for a, b, c in zip(aa, bb, cc)
    ])

    loss.backward()

    gradients = (
        np.array([v.grad for v in aa]),
        np.array([v.grad for v in bb]),
        np.array([v.grad for v in cc])
    )

    return loss.data, gradients


def torch_expression(a_values, b_values, c_values):

    a = torch.tensor(
        a_values, dtype=torch.float64, requires_grad=True
    )

    b = torch.tensor(
        b_values, dtype=torch.float64, requires_grad=True
    )

    c = torch.tensor(
        c_values, dtype=torch.float64, requires_grad=True
    )

    loss = torch.relu((a * b + c) ** 2).sum()

    loss.backward()

    return loss.item(), (
        a.grad.numpy(),
        b.grad.numpy(),
        c.grad.numpy()
    )


def numerical_gradients(a_values, b_values, c_values, eps=1e-6):

    arrays = [
        np.array(a_values, dtype=float),
        np.array(b_values, dtype=float),
        np.array(c_values, dtype=float)
    ]

    results = []

    for k, arr in enumerate(arrays):

        grad = np.zeros_like(arr)

        for i in range(len(arr)):

            plus = [v.copy() for v in arrays]
            minus = [v.copy() for v in arrays]

            plus[k][i] += eps
            minus[k][i] -= eps

            def fn(parts):
                aa, bb, cc = parts
                return np.maximum(
                    (aa * bb + cc) ** 2, 0
                ).sum()

            grad[i] = (
                fn(plus) - fn(minus)
            ) / (2 * eps)

        results.append(grad)

    return tuple(results)


a0 = np.array([0.5, -1.2, 2.0, 0.7])
b0 = np.array([1.3, 0.8, -0.4, 1.1])
c0 = np.array([0.2, 0.5, 1.0, -0.1])

custom_loss, custom_grads = custom_expression(a0, b0, c0)

torch_loss, torch_grads = torch_expression(a0, b0, c0)

finite_grads = numerical_gradients(a0, b0, c0)

differences = []

for cg, tg, ng in zip(
    custom_grads, torch_grads, finite_grads
):

    differences.extend([
        np.max(np.abs(cg - tg)),
        np.max(np.abs(cg - ng)),
        np.max(np.abs(tg - ng))
    ])

max_difference = max(differences)

print("Custom loss:", custom_loss)
print("PyTorch loss:", torch_loss)

print("Maximum absolute gradient difference:", max_difference)

assert max_difference < 1e-6

print("Task 4 passed.")


# ============================================================
# TASK 5: TRAIN MODEL ON MAKE_MOONS
# Architecture: 2 -> 8 -> 1
# ============================================================

X, y = make_moons(
    n_samples=160,
    noise=0.20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

EPOCHS = 12
LR = 0.35
RUNS = 5


def sigmoid_value(z):
    return (1.0 + (-z).exp()) ** -1


def init_custom(seed):

    rng = np.random.default_rng(seed)

    W1 = [
        [Value(rng.normal(0, np.sqrt(2/10)))
         for _ in range(8)]
        for _ in range(2)
    ]

    b1 = [Value(0.0) for _ in range(8)]

    W2 = [
        Value(rng.normal(0, np.sqrt(2/9)))
        for _ in range(8)
    ]

    b2 = Value(0.0)

    return W1, b1, W2, b2


def custom_forward(x, params):

    W1, b1, W2, b2 = params

    hidden = []

    for j in range(8):

        z = b1[j] + sum(
            (float(x[i]) * W1[i][j] for i in range(2)),
            Value(0.0)
        )

        hidden.append(z.relu())

    logit = b2 + sum(
        (hidden[j] * W2[j] for j in range(8)),
        Value(0.0)
    )

    return sigmoid_value(logit)


def train_custom(seed):

    params = init_custom(seed)

    W1, b1, W2, b2 = params

    for epoch in range(EPOCHS):

        predictions = [
            custom_forward(x, params)
            for x in X_train
        ]

        eps = 1e-8

        losses = [
            -(
                float(target) * (p + eps).log()
                +
                (1.0 - float(target))
                * (1.0 - p + eps).log()
            )
            for p, target in zip(predictions, y_train)
        ]

        loss = value_sum(losses) / len(losses)

        loss.backward()

        for row in W1:
            for p in row:
                p.data -= LR * p.grad
                p.grad = 0.0

        for p in b1:
            p.data -= LR * p.grad
            p.grad = 0.0

        for p in W2:
            p.data -= LR * p.grad
            p.grad = 0.0

        b2.data -= LR * b2.grad
        b2.grad = 0.0

    return params


def custom_run(seed):

    start = time.perf_counter()

    params = train_custom(seed)

    elapsed = time.perf_counter() - start

    probs = np.array([
        custom_forward(x, params).data
        for x in X_test
    ])

    accuracy = (
        (probs >= 0.5).astype(int) == y_test
    ).mean()

    return elapsed, accuracy


class MoonNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def torch_run(seed):

    torch.manual_seed(seed)

    model = MoonNet()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR
    )

    criterion = nn.BCELoss()

    xt = torch.tensor(
        X_train, dtype=torch.float32
    )

    yt = torch.tensor(
        y_train.reshape(-1, 1),
        dtype=torch.float32
    )

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        optimizer.zero_grad(set_to_none=True)

        predictions = model(xt)

        loss = criterion(predictions, yt)

        loss.backward()

        optimizer.step()

    elapsed = time.perf_counter() - start

    with torch.no_grad():

        probs = model(
            torch.tensor(X_test, dtype=torch.float32)
        ).numpy().ravel()

    accuracy = (
        (probs >= 0.5).astype(int) == y_test
    ).mean()

    return elapsed, accuracy


custom_results = [
    custom_run(100 + i)
    for i in range(RUNS)
]

torch_results = [
    torch_run(100 + i)
    for i in range(RUNS)
]

custom_times = [r[0] for r in custom_results]
torch_times = [r[0] for r in torch_results]

print("\nTASK 5 RESULTS")

print("Custom engine times:", custom_times)
print("PyTorch times:", torch_times)

print(
    "Custom median:",
    statistics.median(custom_times)
)

print(
    "PyTorch median:",
    statistics.median(torch_times)
)

print(
    "Custom median accuracy:",
    statistics.median([r[1] for r in custom_results])
)

print(
    "PyTorch median accuracy:",
    statistics.median([r[1] for r in torch_results])
)


# ============================================================
# TASK 6: GRAPH, EXECUTION AND TENSOR CONTIGUITY
# ============================================================

torch.set_num_threads(1)

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
).eval()

example = torch.randn(16, 4)

traced = torch.jit.trace(model, example)

print("\nEAGER MODEL")
print(model)

print("\nTRACED COMPUTATION GRAPH")
print(traced.graph)


def median_ms_per_call(
    fn,
    x,
    warmup=20,
    repeats=5,
    iterations=100
):

    with torch.no_grad():

        for _ in range(warmup):
            fn(x)

        samples = []

        for _ in range(repeats):

            start = time.perf_counter()

            for _ in range(iterations):
                fn(x)

            elapsed = time.perf_counter() - start

            samples.append(
                elapsed * 1000 / iterations
            )

    return statistics.median(samples), samples


eager_median, eager_samples = median_ms_per_call(
    model, example
)

traced_median, traced_samples = median_ms_per_call(
    traced, example
)

print("\nEXECUTION TIMINGS")

print("Eager samples (ms):", eager_samples)
print("Traced samples (ms):", traced_samples)

print("Eager median (ms):", eager_median)
print("Traced median (ms):", traced_median)


# Tensor shape, dtype and contiguity

base = torch.randn(512, 1024, dtype=torch.float32)

noncontiguous = base[:, ::2]

contiguous = noncontiguous.contiguous()

print("\nTENSOR INFORMATION")

for name, tensor in [
    ("Base", base),
    ("Non-contiguous view", noncontiguous),
    ("Contiguous copy", contiguous)
]:

    print(
        name,
        "| Shape:", tuple(tensor.shape),
        "| Dtype:", tensor.dtype,
        "| Stride:", tensor.stride(),
        "| Contiguous:", tensor.is_contiguous()
    )


# Compare operation speed

weight = torch.randn(512, 128)


def matmul_op(tensor):
    return tensor @ weight


noncontig_median, noncontig_samples = median_ms_per_call(
    matmul_op,
    noncontiguous,
    warmup=10,
    repeats=5,
    iterations=50
)

contig_median, contig_samples = median_ms_per_call(
    matmul_op,
    contiguous,
    warmup=10,
    repeats=5,
    iterations=50
)

print("\nCONTIGUITY PERFORMANCE")

print(
    "Non-contiguous times (ms):",
    noncontig_samples
)

print(
    "Contiguous times (ms):",
    contig_samples
)

print(
    "Non-contiguous median (ms):",
    noncontig_median
)

print(
    "Contiguous median (ms):",
    contig_median
)


# ============================================================
# END OF LAB 5
# ============================================================

print("\nAll six lab tasks have finished running.")
