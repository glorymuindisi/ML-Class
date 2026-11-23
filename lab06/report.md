# Individual Project Proposal: Lightweight Edge Book Recommendation Engine

* **Student Name:** Glory Tendaishe Muindisi
* **Student ID:** 70642


### 1. Problem Statement
Digital reading platforms and offline e-reader applications require fast, personalized book recommendations without relying on cloud servers or heavy memory footprints. The objective of this project is to build a compact book recommendation model capable of predicting user preferences and suggesting relevant books locally on resource-constrained edge or mobile hardware.

### 2. Dataset
* **Dataset Name:** Goodreads / Book-Crossing Interactions Subset
* **Description:** User-book interaction records (ratings, implicit reads) paired with lightweight metadata vectors (author ID, primary genre tags, and normalized page count). Filtered to top active users and popular titles to keep embedding tables compact.
* **Train/Val/Test Split:** 80% training / 10% validation / 10% test (split chronologically or per-user).

### 3. Success Metric
* **Primary Metric:** Hit Rate @ 5 (>= 75%) on the held-out test interactions (measuring how often the user's actual next-read book appears in the top 5 recommendations).
* **Secondary Metric:** Normalized Discounted Cumulative Gain (NDCG@5 >= 0.60).

### 4. Deployment Constraints
* **Latency Constraint:** <= 50 ms per inference query (generating top-K candidate scores for a single user on CPU).
* **Model Size Constraint:** <= 5 MB maximum exported checkpoint/weights size.

### 5. Planned Optimization Strategy
To guarantee the model stays well within the 5 MB storage limit and <= 50 ms latency budget:
1. **Compact Embedding Architecture:** Use a lightweight Two-Tower / Neural Collaborative Filtering (NCF) architecture with small embedding dimensions (d = 16 or 32) and two compact fully connected layers.
2. **Post-Training Quantization (PTQ):** Apply PyTorch dynamic/static quantization (qint8) to the embedding lookup tables and linear layers, reducing model memory footprint by ~4x and accelerating CPU inference.
3. **Pruning (Fallback):** Apply unstructured magnitude pruning to embedding weights if total file size approaches the 5 MB boundary.




# Lab 6 Report: Training Pipelines and Project Proposal

**Student Name:** [Your Full Name]
**Student ID:** [Your Student ID]

## 1. YAML Pipeline, Checkpoints, and MLflow (Task 1)
Training is driven by `lab06/configs/config.yaml` with fixed random seeds (`seed: 42`, `cudnn.deterministic = True`), model/optimizer/scheduler checkpoints saved to `lab06/results/checkpoints/`, and parameters, metrics, and artifacts logged to MLflow in `lab06/results/mlruns`.

## 2. Single-Epoch Profiler Breakdown (Task 2)
Saved in `lab06/results/task2_profiler_breakdown.csv`, `lab06/results/task2_torch_profiler_ops.txt`, and `lab06/results/task2_profiler_breakdown.png`.
- Data Loading: `[copy from task2_profiler_breakdown.csv]`
- Forward Pass: `[copy from task2_profiler_breakdown.csv]`
- Backward Pass: `[copy from task2_profiler_breakdown.csv]`
- Optimizer Step: `[copy from task2_profiler_breakdown.csv]`

## 3. DataLoader num_workers Sweep [0, 1, 2, 4, 8] (Task 3)
Saved in `lab06/results/task3_num_workers.csv` and `lab06/results/task3_num_workers.png`.
On Windows, process spawning and IPC overhead affect throughput as worker count increases; the CSV and plot identify the fastest worker setting and bottleneck.

## 4. Batch-Size Study [32, 64, 128, 256, 512] — 5 Epochs (Task 4)
Saved in `lab06/results/task4_batch_size_study.csv` and `lab06/results/task4_batch_size_study.png`.
Records throughput (samples/s), peak memory (MB), and 5-epoch validation accuracy across batch sizes.

## 5. Optimizers and LR Schedulers — 10 Epochs (Task 5)
Saved in `lab06/results/task5_optim_scheduler_comparison.csv` and `lab06/results/task5_optim_scheduler_comparison.png`.
Compares SGD+Momentum and Adam across `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` over equal 10-epoch budgets.

## 6. Gradient Accumulation: Batch 256 vs. Batch 64 x 4 (Task 6)
Saved in `lab06/results/task6_grad_accumulation.csv` and `lab06/results/task6_grad_accumulation.png`.
Scaling loss by `1/4` over 4 micro-batches of 64 emulates effective batch size 256 with lower peak activation memory and similar loss/accuracy curves.

## 7. Checkpoint Interruption at Epoch 3 and Resume (Task 7)
Saved in `lab06/results/task7_resume_curve.csv`, `lab06/results/task7_resume_curve.png`, and `lab06/results/checkpoints/task7_resumed_epoch_3.pt`.
Training halts at Epoch 3 and resumes from the saved checkpoint with restored model, optimizer, and scheduler states, maintaining smooth loss/accuracy curves.

## 8. Individual Project Proposal (One Page)
**Student Name:** [Your Full Name]  
**Student ID:** [Your Student ID]  
**Problem:** Real-time image classification on edge hardware with strict memory and latency limits.  
**Dataset:** Fashion-MNIST / CIFAR-10 (or NEU Surface Defect Dataset), split into train/validation/test sets with standard normalization and light augmentation.  
**Success Metric:** Top-1 test accuracy >= 90% and macro F1 >= 0.89 while meeting latency and model size limits.  
**Deployment Constraints:** Single-sample CPU latency <= 50 ms and serialized model size <= 5 MB.  
**Planned Optimization:** Structured channel pruning followed by fine-tuning and INT8 static quantization (`torch.ao.quantization`) to reduce model footprint and speed up CPU execution, evaluated against the unpruned FP32 baseline.