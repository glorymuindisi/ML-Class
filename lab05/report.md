# Lab 5: Frameworks and automatic differentiation

Environment: Python 3.13.18, PyTorch 2.15.0+cu132, Linux-6.18.44-fc-v726-x86_64-with-glibc2.39, 1 CPU thread in Task 6

All numbers below come from `results/output.txt` (the unedited output of `src/frameworks.py`); `results/summary.json` and the two CSV files hold the same data.

## 1-3. Value node, operations and backward pass

`Value` stores `data`, `grad`, its parent nodes (`_prev`) and a local derivative rule (`_backward`). Addition, multiplication, powers and ReLU each create a new node in the forward pass, so the computation graph is built as the expression is evaluated. `backward()` topologically sorts the graph, sets the output gradient to 1, visits nodes in reverse order, and accumulates gradients with `+=` (never overwrites).

Accumulation check: for `f = x*x + x` at `x = 3`, `x` is used three times and the gradient is `2x + 1 = 7`. The engine returned 7.0, so contributions are summed correctly.

## 4. Gradient verification

Expression: `((a*b + c)**2).relu().sum()` with four values each for `a`, `b`, `c`.

| Quantity | Value |
|---|---|
| Loss, own engine | 1.4230000000000003 |
| Loss, PyTorch | 1.4230000000000003 |
| Max absolute gradient difference (engine vs PyTorch vs numerical) | 1.176e-10 |
| Required | < 1e-6 |
| Result | PASS |

The reported difference is the maximum over all three pairwise comparisons (own engine vs PyTorch, own engine vs numerical, PyTorch vs numerical).

## 5. Training on make_moons (2 -> 8 -> 1, sigmoid, BCE)

Setup: 160 samples (120 train, 40 test), noise 0.20, standardized inputs, 12 epochs of full-batch gradient descent, learning rate 0.35, 5 runs with seeds 100-104.

| Run | Own engine (s) | PyTorch (s) |
|---|---|---|
| 1 | 1.0823 | 1.0489 |
| 2 | 0.9614 | 0.0023 |
| 3 | 0.8795 | 0.0023 |
| 4 | 1.2010 | 0.0024 |
| 5 | 1.1770 | 0.0023 |
| **Median** | **1.0823** | **0.00230** |

| | Own engine | PyTorch |
|---|---|---|
| Median test accuracy | 0.900 | 0.875 |

PyTorch is about **470x faster** (median 1.082 s vs 0.002301 s). The own engine builds and walks a graph of individual Python objects, with one node per scalar operation, while PyTorch runs a few vectorized kernels per step.

Notes on reading these numbers:

- PyTorch run 1 (1.049 s) is much slower than runs 2-5 (about 0.0023 s). That first run pays one-time start-up costs (library initialization). The median ignores it; reporting the mean would not.
- Test accuracy is measured on 40 points, so one sample is 2.5 percentage points. The two implementations use different random initializations (0.900 vs 0.875), so this difference is within noise and does not show one is more accurate.
- Only 12 epochs are trained, so the models are not converged; the lab compares speed, not final accuracy.

## 6. PyTorch graph, eager vs traced execution, tensor layout

### Graph of a small model (4 -> 8 -> 2)

```
Sequential(
  (0): Linear(in_features=4, out_features=8, bias=True)
  (1): ReLU()
  (2): Linear(in_features=8, out_features=2, bias=True)
)
```

`torch.jit.trace` graph:

```
graph(%self.1 : __torch__.torch.nn.modules.container.Sequential,
      %input.1 : Float(16, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %_2 : __torch__.torch.nn.modules.linear.___torch_mangle_0.Linear = prim::GetAttr[name="2"](%self.1)
  %_1 : __torch__.torch.nn.modules.activation.ReLU = prim::GetAttr[name="1"](%self.1)
  %_0 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="0"](%self.1)
  %33 : Tensor = prim::CallMethod[name="forward"](%_0, %input.1)
  %34 : Tensor = prim::CallMethod[name="forward"](%_1, %33)
  %35 : Tensor = prim::CallMethod[name="forward"](%_2, %34)
  return (%35)
```

The graph is a short chain: Linear, ReLU, Linear. Each module call appears as a `prim::CallMethod`, and the input is recorded as `Float(16, 4)` with strides `[4, 1]`.

### Eager vs traced (after 20 warm-up calls, median of 5 repeats of 100 calls)

| Mode | Median ms per call |
|---|---|
| Eager | 0.01101 |
| `torch.jit.trace` | 0.00970 |

The traced model is 1.14x faster than eager. `torch.jit.trace` was used (the lab allows tracing as the fallback); `torch.compile` was not measured.

### Shape, dtype and contiguity

| Tensor | Shape | Dtype | Stride | Contiguous |
|---|---|---|---|---|
| Base | (512, 1024) | torch.float32 | (1024, 1) | True |
| Non-contiguous view | (512, 512) | torch.float32 | (1024, 2) | False |
| Contiguous copy | (512, 512) | torch.float32 | (512, 1) | True |

`base[:, ::2]` is a view of the same memory with stride `(1024, 2)`: it skips every other element, so consecutive elements of a row are not adjacent in memory. `.contiguous()` copies it into a dense layout with stride `(512, 1)`.

### Cost of a non-contiguous view (512x512 @ 512x128 matmul)

| Input | Median ms per call |
|---|---|
| Non-contiguous view | 0.6891 |
| Contiguous copy | 0.4880 |

The non-contiguous input was **1.41x** slower. Strided reads waste memory bandwidth and cache lines, and matrix-multiply libraries are tuned for dense layouts. Calling `.contiguous()` costs one copy, which typically pays off when the same tensor is used in several operations.

## When does compilation help?

Compilation and tracing remove per-operation framework overhead and can fuse several small operations into one kernel, which saves memory traffic. That matters when a model is made of many small element-wise operations, when it runs many times so the one-off compile or trace cost is paid back, and when it runs on a GPU where launch overhead is large. For this tiny model (two small matrix multiplies, about 11 microseconds per call) the gain was only 1.14x, because there is very little overhead to remove and almost nothing to fuse. For large models dominated by big matrix multiplies, eager mode already calls optimized kernels, so compilation gives a smaller relative gain there too.