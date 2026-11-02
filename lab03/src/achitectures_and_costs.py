# Install: python -m pip install torch torchvision torchinfo thop matplotlib
import time
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from thop import profile

#Task 1

torch.manual_seed(0)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
test = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
train, val = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(0))

train_dl = DataLoader(train, batch_size=128, shuffle=True)
val_dl = DataLoader(val, batch_size=128)
test_dl = DataLoader(test, batch_size=128)

#task 2

mlp = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(49, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(64, 10)

    def forward(self, x):                                   # x: (B, 1, 28, 28)
        p = x.unfold(2, 7, 7).unfold(3, 7, 7)               # (B, 1, 4, 4, 7, 7)
        p = p.reshape(x.size(0), 16, 49)                    # 16 patches of 49 pixels
        return self.head(self.encoder(self.proj(p)).mean(dim=1))

attn = PatchAttention()
models = {"MLP": mlp, "CNN": cnn, "Attention": attn}

#task 3a
for name, m in models.items():
    print(name)
    summary(m, input_size=(1, 1, 28, 28))


#   Linear 784->256:                    2*784*256                      = 401,408
#   Conv2d 16->32, 3x3, 14x14 output:   2*3*3*16*32*14*14              = 1,806,336
#   Attention (seq 16, d_model 64):
#     QKV projections                   2*16*64*(3*64)                 = 393,216
#     scores                            2*16^2*64                      = 32,768
#     weighted sum                      2*16^2*64                      = 32,768
#     output projection                 2*16*64*64                     = 131,072
#     feed-forward (64->128->64)        2*16*64*128*2                  = 524,288
#     full encoder layer                                               = 1,114,112


# Task 3c
x1 = torch.randn(1, 1, 28, 28)
for name, m in models.items():
    macs, p = profile(m, inputs=(x1,), verbose=False)
    print(f"{name}: thop MACs={macs:,.0f}  (x2 = {2 * macs:,.0f} FLOPs)  params={p:,.0f}")

def latency_ms(m, batch, runs):
    m.eval(); x = torch.randn(batch, 1, 28, 28); t = []
    with torch.no_grad():
        for _ in range(5): m(x)                      # warm-up
        for _ in range(runs):
            s = time.perf_counter(); m(x); t.append(time.perf_counter() - s)
    return np.median(t) * 1e3

lat = {n: (latency_ms(m, 1, 100), latency_ms(m, 128, 20)) for n, m in models.items()}
for n, (a, b) in lat.items():
    print(f"{n}: batch 1 {a:.3f} ms, batch 128 {b:.3f} ms")

#task 4
def accuracy(m, dl):
    m.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in dl:
            ok += (m(x).argmax(1) == y).sum().item(); n += len(y)
    return ok / n

test_acc = {}
for name, m in models.items():
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); loss_fn = nn.CrossEntropyLoss()
    for ep in range(5):
        m.train()
        for x, y in train_dl:
            opt.zero_grad(); loss_fn(m(x), y).backward(); opt.step()
        print(f"{name} epoch {ep + 1} val acc {accuracy(m, val_dl):.4f}")
    test_acc[name] = accuracy(m, test_dl)
    print(f"{name} TEST accuracy {test_acc[name]:.4f}")

#task5
params = {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in models:
    ax[0].scatter(params[n], test_acc[n], s=80); ax[0].annotate(n, (params[n], test_acc[n]))
    ax[1].scatter(lat[n][0], test_acc[n], marker="o", s=80, label=f"{n} (batch 1)")
    ax[1].scatter(lat[n][1], test_acc[n], marker="s", s=80, label=f"{n} (batch 128)")
ax[0].set(xlabel="Parameters", ylabel="Test accuracy", title="Accuracy vs parameters")
ax[1].set(xlabel="Latency (ms, median)", ylabel="Test accuracy", title="Accuracy vs latency", xscale="log")
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("tradeoffs.png", dpi=150)

