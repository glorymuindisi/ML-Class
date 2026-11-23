import os
import json
import time
import copy
import random
import argparse
from pathlib import Path
from contextlib import nullcontext

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
import torchvision
import torchvision.transforms as T

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lab3BestCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, hidden_channels=(32, 64), dropout=0.25):
        super().__init__()
        c1, c2 = hidden_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_datasets(cfg):
    ds_name = cfg["dataset"]["name"].lower()
    data_dir = cfg["dataset"]["data_dir"]
    if ds_name == "cifar10":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
        train_ds = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_ds, val_ds


def make_loaders(train_ds, val_ds, batch_size, num_workers, pin_memory, device):
    use_pin = bool(pin_memory and device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    return train_loader, val_loader


def build_optimizer(model, cfg):
    opt_name = cfg["training"]["optimizer"].lower()
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=float(cfg["training"]["momentum"]), weight_decay=wd)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg):
    sched_name = cfg["training"]["scheduler"].lower()
    epochs = int(cfg["training"]["epochs"])
    if sched_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg["training"]["step_size"]),
            gamma=float(cfg["training"]["gamma"]),
        )
    elif sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif sched_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    return None


def save_checkpoint(path, epoch, model, optimizer, scheduler, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"]), ckpt.get("history", [])


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), 100.0 * correct / max(1, total)


def train_experiment(cfg, train_ds, val_ds, run_name="baseline", save_ckpts=True):
    set_seed(int(cfg["seed"]))
    device = get_device()
    results_dir = Path(cfg["results_dir"])
    ckpt_dir = Path(cfg["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["dataset"]["num_workers"])
    accum_steps = max(1, int(cfg["training"]["accumulation_steps"]))
    epochs = int(cfg["training"]["epochs"])
    interrupt_epoch = cfg["training"].get("interrupt_at_epoch")
    resume_path = cfg["training"].get("resume_from")

    train_loader, val_loader = make_loaders(
        train_ds, val_ds, batch_size, num_workers, cfg["dataset"]["pin_memory"], device
    )

    model = Lab3BestCNN(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    history = []
    if resume_path and os.path.exists(resume_path):
        last_ep, history = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = last_ep + 1

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
        mlflow.set_experiment(cfg["mlflow_experiment_name"])
        mlflow_ctx = mlflow.start_run(run_name=run_name)
    else:
        mlflow_ctx = nullcontext()

    with mlflow_ctx:
        if HAS_MLFLOW:
            mlflow.log_params({
                "seed": cfg["seed"],
                "batch_size": batch_size,
                "accumulation_steps": accum_steps,
                "effective_batch_size": batch_size * accum_steps,
                "num_workers": num_workers,
                "optimizer": cfg["training"]["optimizer"],
                "lr": cfg["training"]["lr"],
                "scheduler": cfg["training"]["scheduler"],
                "epochs": epochs,
                "device": str(device),
            })

        peak_mem_mb = 0.0
        throughputs = []

        for epoch in range(start_epoch, epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            model.train()
            optimizer.zero_grad()
            running_loss, correct, total = 0.0, 0, 0
            t0 = time.perf_counter()

            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                raw_loss = criterion(out, y)
                loss = raw_loss / accum_steps
                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += raw_loss.item() * y.size(0)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
                ep_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_mem_mb = max(peak_mem_mb, ep_mem)

            dt = max(1e-6, time.perf_counter() - t0)
            throughput = total / dt
            throughputs.append(throughput)

            train_loss = running_loss / max(1, total)
            train_acc = 100.0 * correct / max(1, total)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
                "lr": current_lr,
                "epoch_time_s": round(dt, 3),
                "throughput_samples_s": round(throughput, 1),
            })

            if HAS_MLFLOW:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "throughput_samples_s": throughput,
                }, step=epoch)

            if save_ckpts:
                ckpt_file = ckpt_dir / f"{run_name}_epoch_{epoch}.pt"
                save_checkpoint(str(ckpt_file), epoch, model, optimizer, scheduler, history)

            if interrupt_epoch is not None and epoch == int(interrupt_epoch):
                break

        summary = {
            "run_name": run_name,
            "final_val_acc": history[-1]["val_acc"] if history else 0.0,
            "final_val_loss": history[-1]["val_loss"] if history else 0.0,
            "avg_throughput_samples_s": round(float(np.mean(throughputs)), 1) if throughputs else 0.0,
            "peak_gpu_mem_mb": round(peak_mem_mb, 2),
            "history": history,
        }

        hist_path = results_dir / f"{run_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump(summary, f, indent=2)

        if HAS_MLFLOW:
            mlflow.log_artifact(str(hist_path))

    return summary


def run_task2_profiler(cfg, train_ds, val_ds):
    device = get_device()
    results_dir = Path(cfg["results_dir"])
    train_loader, _ = make_loaders(
        train_ds, val_ds,
        int(cfg["training"]["batch_size"]),
        int(cfg["dataset"]["num_workers"]),
        cfg["dataset"]["pin_memory"],
        device,
    )
    model = Lab3BestCNN(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    t_data, t_fwd, t_bwd, t_opt = 0.0, 0.0, 0.0, 0.0
    model.train()

    with profile(activities=activities, record_shapes=True) as prof:
        it = iter(train_loader)
        while True:
            t0 = time.perf_counter()
            try:
                x, y = next(it)
            except StopIteration:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()

            t_data += (t1 - t0)
            t_fwd += (t2 - t1)
            t_bwd += (t3 - t2)
            t_opt += (t4 - t3)

    total = max(1e-6, t_data + t_fwd + t_bwd + t_opt)
    breakdown = pd.DataFrame([
        {"Stage": "Data Loading", "Time_s": round(t_data, 4), "Share_pct": round(100 * t_data / total, 2)},
        {"Stage": "Forward Pass", "Time_s": round(t_fwd, 4), "Share_pct": round(100 * t_fwd / total, 2)},
        {"Stage": "Backward Pass", "Time_s": round(t_bwd, 4), "Share_pct": round(100 * t_bwd / total, 2)},
        {"Stage": "Optimizer Step", "Time_s": round(t_opt, 4), "Share_pct": round(100 * t_opt / total, 2)},
    ])
    breakdown.to_csv(results_dir / "task2_profiler_breakdown.csv", index=False)

    with open(results_dir / "task2_torch_profiler_ops.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(breakdown["Stage"], breakdown["Share_pct"], color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"])
    ax.set_ylabel("Share of Epoch Time (%)")
    ax.set_title("Task 2: Single-Epoch Time Breakdown")
    for idx, val in enumerate(breakdown["Share_pct"]):
        ax.text(idx, val + 0.8, f"{val:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(results_dir / "task2_profiler_breakdown.png", dpi=150)
    plt.close(fig)


def run_task3_workers(cfg, train_ds, val_ds):
    device = get_device()
    results_dir = Path(cfg["results_dir"])
    rows = []

    for nw in cfg["tasks"]["worker_candidates"]:
        train_loader, _ = make_loaders(
            train_ds, val_ds,
            int(cfg["training"]["batch_size"]),
            int(nw),
            cfg["dataset"]["pin_memory"],
            device,
        )
        t0 = time.perf_counter()
        samples = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            samples += y.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = max(1e-6, time.perf_counter() - t0)
        rows.append({
            "num_workers": nw,
            "epoch_load_time_s": round(dt, 3),
            "throughput_samples_s": round(samples / dt, 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "task3_num_workers.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["num_workers"], df["throughput_samples_s"], marker="o", linewidth=2, color="#2B5C8F")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Data Loading Throughput (samples/s)")
    ax.set_title("Task 3: DataLoader Throughput vs. num_workers")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "task3_num_workers.png", dpi=150)
    plt.close(fig)


def run_task4_batch_study(cfg, train_ds, val_ds):
    results_dir = Path(cfg["results_dir"])
    rows = []

    for bs in cfg["tasks"]["batch_candidates"]:
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["training"]["batch_size"] = int(bs)
        sub_cfg["training"]["epochs"] = 5
        summary = train_experiment(sub_cfg, train_ds, val_ds, run_name=f"task4_bs_{bs}", save_ckpts=False)
        rows.append({
            "batch_size": bs,
            "avg_throughput_samples_s": summary["avg_throughput_samples_s"],
            "peak_gpu_mem_mb": summary["peak_gpu_mem_mb"],
            "final_val_acc_pct": summary["final_val_acc"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "task4_batch_size_study.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["batch_size"], df["avg_throughput_samples_s"], marker="o", color="#2B5C8F")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Throughput (samples/s)")
    axes[0].set_title("Throughput vs. Batch Size")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(df["batch_size"], df["final_val_acc_pct"], marker="s", color="#D95F02")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Final Val Accuracy (%)")
    axes[1].set_title("5-Epoch Accuracy vs. Batch Size")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_dir / "task4_batch_size_study.png", dpi=150)
    plt.close(fig)


def run_task5_optim_sched(cfg, train_ds, val_ds):
    results_dir = Path(cfg["results_dir"])
    combos = [
        ("sgd", 0.01, "step"),
        ("sgd", 0.01, "cosine"),
        ("sgd", 0.01, "plateau"),
        ("adam", 0.001, "step"),
        ("adam", 0.001, "cosine"),
        ("adam", 0.001, "plateau"),
    ]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for opt_name, lr, sched_name in combos:
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["training"]["epochs"] = 10
        sub_cfg["training"]["optimizer"] = opt_name
        sub_cfg["training"]["lr"] = lr
        sub_cfg["training"]["scheduler"] = sched_name
        label = f"{opt_name.upper()}_{sched_name}"
        summary = train_experiment(sub_cfg, train_ds, val_ds, run_name=f"task5_{label}", save_ckpts=False)
        hist_df = pd.DataFrame(summary["history"])
        ax.plot(hist_df["epoch"], hist_df["val_acc"], marker="o", label=label)
        rows.append({
            "optimizer": opt_name.upper(),
            "scheduler": sched_name,
            "initial_lr": lr,
            "final_val_acc_pct": summary["final_val_acc"],
            "final_val_loss": summary["final_val_loss"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "task5_optim_scheduler_comparison.csv", index=False)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Task 5: SGD+Momentum vs. Adam Across LR Schedulers (10 Epochs)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "task5_optim_scheduler_comparison.png", dpi=150)
    plt.close(fig)


def run_task6_grad_accum(cfg, train_ds, val_ds):
    results_dir = Path(cfg["results_dir"])

    cfg_256 = copy.deepcopy(cfg)
    cfg_256["training"]["batch_size"] = 256
    cfg_256["training"]["accumulation_steps"] = 1
    cfg_256["training"]["epochs"] = 5
    s_256 = train_experiment(cfg_256, train_ds, val_ds, run_name="task6_bs256", save_ckpts=False)

    cfg_acc = copy.deepcopy(cfg)
    cfg_acc["training"]["batch_size"] = 64
    cfg_acc["training"]["accumulation_steps"] = 4
    cfg_acc["training"]["epochs"] = 5
    s_acc = train_experiment(cfg_acc, train_ds, val_ds, run_name="task6_bs64_accum4", save_ckpts=False)

    h1 = pd.DataFrame(s_256["history"])
    h2 = pd.DataFrame(s_acc["history"])

    comp = pd.DataFrame({
        "epoch": h1["epoch"],
        "bs256_train_loss": h1["train_loss"],
        "bs64x4_train_loss": h2["train_loss"],
        "bs256_val_acc": h1["val_acc"],
        "bs64x4_val_acc": h2["val_acc"],
    })
    comp.to_csv(results_dir / "task6_grad_accumulation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h1["epoch"], h1["train_loss"], marker="o", label="Batch 256 (accum=1)")
    axes[0].plot(h2["epoch"], h2["train_loss"], marker="s", linestyle="--", label="Batch 64 (accum=4)")
    axes[0].set_title("Training Loss: Batch 256 vs. 64x4")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(h1["epoch"], h1["val_acc"], marker="o", label="Batch 256 (accum=1)")
    axes[1].plot(h2["epoch"], h2["val_acc"], marker="s", linestyle="--", label="Batch 64 (accum=4)")
    axes[1].set_title("Validation Accuracy: Batch 256 vs. 64x4")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_dir / "task6_grad_accumulation.png", dpi=150)
    plt.close(fig)


def run_task7_resume(cfg, train_ds, val_ds):
    results_dir = Path(cfg["results_dir"])
    ckpt_dir = Path(cfg["checkpoint_dir"])

    cfg_part1 = copy.deepcopy(cfg)
    cfg_part1["training"]["epochs"] = 5
    cfg_part1["training"]["interrupt_at_epoch"] = 3
    cfg_part1["training"]["resume_from"] = None
    train_experiment(cfg_part1, train_ds, val_ds, run_name="task7_resumed", save_ckpts=True)

    cfg_part2 = copy.deepcopy(cfg)
    cfg_part2["training"]["epochs"] = 5
    cfg_part2["training"]["interrupt_at_epoch"] = None
    cfg_part2["training"]["resume_from"] = str(ckpt_dir / "task7_resumed_epoch_3.pt")
    s_full = train_experiment(cfg_part2, train_ds, val_ds, run_name="task7_resumed", save_ckpts=True)

    hist_df = pd.DataFrame(s_full["history"])
    hist_df.to_csv(results_dir / "task7_resume_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="Train Loss")
    ax.plot(hist_df["epoch"], hist_df["val_loss"], marker="s", label="Val Loss")
    ax.axvline(x=3, color="red", linestyle="--", label="Interrupted & Resumed (Epoch 3)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Task 7: Smooth Loss Curve Across Epoch-3 Checkpoint Resume")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "task7_resume_curve.png", dpi=150)
    plt.close(fig)


def print_saved_results_summary(results_dir: Path):
    print("\n================ SAVED RESULTS IN lab06/results ================")
    for p in sorted(results_dir.glob("*.csv")):
        print(f"\n--- {p.name} ---")
        print(pd.read_csv(p).to_string(index=False))
    print("\n--- Generated Diagrams (.png) ---")
    for png in sorted(results_dir.glob("*.png")):
        print(" ", png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="lab06/configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    train_ds, val_ds = get_datasets(cfg)

    if cfg["tasks"].get("run_baseline", True):
        train_experiment(cfg, train_ds, val_ds, run_name="task1_baseline", save_ckpts=True)
    if cfg["tasks"].get("run_profiler", True):
        run_task2_profiler(cfg, train_ds, val_ds)
    if cfg["tasks"].get("run_worker_sweep", True):
        run_task3_workers(cfg, train_ds, val_ds)
    if cfg["tasks"].get("run_batch_sweep", True):
        run_task4_batch_study(cfg, train_ds, val_ds)
    if cfg["tasks"].get("run_optim_sched_sweep", True):
        run_task5_optim_sched(cfg, train_ds, val_ds)
    if cfg["tasks"].get("run_grad_accum", True):
        run_task6_grad_accum(cfg, train_ds, val_ds)
    if cfg["tasks"].get("run_resume_demo", True):
        run_task7_resume(cfg, train_ds, val_ds)

    print_saved_results_summary(results_dir)


if __name__ == "__main__":
    main()