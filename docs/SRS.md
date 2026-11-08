# Software Requirements Specification
## Neural Networks From Scratch

Srivalli Nalla

## Table of Contents

1. Introduction
   - 1.1 Purpose
   - 1.2 System Overview
   - 1.3 Design Map
   - 1.4 Definitions and Acronyms
2. Design Considerations
   - 2.1 Assumptions
   - 2.2 Constraints
   - 2.3 System Environment
   - 2.4 Risk and Volatile Areas
3. High Level Design
4. Detailed Design
   - 4.1 Data Sources
   - 4.2 Core Modules and Functions
5. Interface Design
   - 5.1 Digits Training Run
   - 5.2 CIFAR-10 Training Run
   - 5.3 Gradient Check and Unit Tests
   - 5.4 Outputs and Reports
- Appendix A: Project Timeline and Measured Results

## 1. Introduction

### 1.1 Purpose

The purpose of this document is to outline the design and architecture of **Neural Networks From Scratch**, a Python project that implements neural networks at the lowest practical level (hand-derived backpropagation and a hand-written Adam optimizer in NumPy) and then rebuilds the same ideas in PyTorch with custom architectures and a manual training loop. The project exists to demonstrate verifiable, first-principles deep learning skill: every component that a framework would normally hide is implemented, gradient-checked, and measured here.

This application is designed for:

- **The author**, as portfolio evidence of deep learning fundamentals for research supervisors and employers.
- **Reviewers of the author's applications** (for example, PhD supervisors screening for PyTorch and custom-architecture experience).
- **Learners** who want a small, complete, readable example of backpropagation, Adam, and a residual convolutional network without framework magic.

### 1.2 System Overview

The system is a **command-line training application** built using:

- **Python 3.12** (core programming language)
- **NumPy** (the from-scratch network: forward pass, backward pass, Adam)
- **PyTorch 2.x (CPU)** (custom MLP and custom residual CNN, manual training loop)
- **scikit-learn digits dataset** (1,797 handwritten digit images, 8x8)
- **CIFAR-10 via torchvision** (50,000 training and 10,000 test images, 32x32 colour)

#### Key Features:

**1. From-Scratch NumPy Network:**
   a. Manual forward and backward passes with He initialization.
   b. Hand-written Adam optimizer; full-batch training.
   c. Numerical gradient check proving the backpropagation is correct.

**2. PyTorch Custom Models:**
   a. A custom multilayer perceptron written directly on PyTorch tensors.
   b. A custom residual convolutional network (stem plus two hand-built residual blocks).
   c. A manual training loop (no high-level trainer wrappers).

**3. Two Benchmarks:**
   a. Handwritten digits: all three models trained and compared on the same split.
   b. CIFAR-10: the residual CNN trained for 30 epochs with basic augmentation (random crop and horizontal flip).

**4. Checkpointing:**
   a. The CIFAR-10 run saves its state after every epoch and can resume after an interruption without losing completed epochs.

### 1.3 Design Map

The system follows a **two-track parallel architecture**, one track per implementation technology, sharing the same data and evaluation protocol:

**1. NumPy Track:**
   a. `NumpyMLP`: layers, activations, and loss implemented as explicit array operations.
   b. `Adam`: the optimizer as a standalone class with first and second moment estimates.

**2. PyTorch Track:**
   a. `TorchMLP`: the same perceptron design expressed with PyTorch modules.
   b. `CifarResCNN`: a residual network assembled from a custom `ResidualBlock`.

**3. Training Scripts:**
   a. One script per dataset; each loads data, trains every model for that dataset, prints progress, and writes a JSON results file.

**4. Output:**
   a. Console progress lines per epoch.
   b. A JSON results file per dataset with per-epoch history and final test accuracy.

### 1.4 Definitions and Acronyms

| Term | Definition |
|---|---|
| Backpropagation | Computing gradients by applying the chain rule layer by layer from the loss backwards. |
| Gradient check | Comparing analytical gradients against numerical (finite-difference) gradients to prove correctness. |
| Adam | Adaptive optimization algorithm using first and second moment estimates of the gradient. |
| He initialization | Weight initialization scaled for ReLU-type activations. |
| Residual block | A network block whose output is added to its input (skip connection), easing training of deeper networks. |
| Epoch | One full pass over the training set. |
| MLP | Multilayer perceptron, a fully connected feed-forward network. |
| CNN | Convolutional neural network. |

## 2. Design Considerations

The project was built in a deliberate order: the NumPy network and its gradient check first (correctness before speed), the PyTorch models second (same designs, framework idioms), the digits benchmark third (fast iteration), and the CIFAR-10 run last (the long, checkpointed run). The main design problems were proving the hand-written gradients correct and making a multi-hour CPU training run survive interruptions.

### 2.1 Assumptions

- The user runs the project inside its bundled virtual environment (`.venv`) with PyTorch, NumPy, scikit-learn, and torchvision installed from `requirements.txt`.
- The digits dataset ships with scikit-learn and needs no download; CIFAR-10 is downloaded once by torchvision on the first run (about 170 MB) and cached under `data/`.
- Training is CPU-only; no GPU is assumed anywhere in the design or the reported timings.

### 2.2 Constraints

**Framework Constraints (by design)**

- The NumPy track may not use any autodiff or framework optimizer; every gradient is hand-derived.
- The PyTorch track uses a manual training loop; high-level training wrappers are out of scope.

**Compute Limitations**

- CPU-only training bounds model size and epochs: the CIFAR-10 configuration (30 epochs) takes roughly 4 to 6 hours on a 4-CPU virtual machine.
- Full-batch training in the NumPy track is chosen for gradient-check comparability, not speed.

**Time Limitations**

- The CIFAR-10 run is long enough to be interrupted by machine restarts; checkpointing after every epoch is a requirement, not an optimization.

### 2.3 System Environment

Python 3.12 on Linux, inside the project virtual environment. Development and all reported runs were performed on a virtual machine with 4 CPUs and about 7 GB of RAM. No API keys, accounts, or network services are needed after the one-time CIFAR-10 download.

### 2.4 Risk and Volatile Areas

**Silent Gradient Bugs**

- A wrong hand-derived gradient still trains, just badly, so it can hide. Mitigation: a numerical gradient check runs in the test suite and must pass before any result is trusted.

**Long-Run Interruption**

- A multi-hour training run can be killed by a machine replacement. Mitigation: per-epoch checkpoints plus a resume path; the reported CIFAR-10 run in fact survived interruptions this way.

**Result Integrity**

- Reported accuracies come only from the JSON files the training scripts write; the README and this document quote those files.

## 3. High Level Design

A training run proceeds in four stages:

1. The script loads the dataset and applies the fixed split (digits: stratified 80/20, seed 0; CIFAR-10: the standard train/test split).
2. Each model is constructed and trained; progress (loss, and test accuracy where applicable) is printed per epoch or per logging interval.
3. For CIFAR-10, the model and history are checkpointed after every epoch.
4. At the end, the script writes the results JSON with the full per-epoch history and the final test accuracy for each model.

## 4. Detailed Design

### 4.1 Data Sources

- **scikit-learn digits**: 1,797 images of 8x8 pixels with labels 0–9; split into 1,437 training and 360 test images.
- **CIFAR-10 (torchvision)**: 50,000 training and 10,000 test colour images across 10 classes, with random-crop and horizontal-flip augmentation during training.

### 4.2 Core Modules and Functions

| Module | Responsibility |
|---|---|
| `src/numpy_net.py` | `NumpyMLP` (manual forward/backward, He initialization) and the hand-written `Adam` optimizer. |
| `src/torch_net.py` | `TorchMLP`, the custom `ResidualBlock`, and `CifarResCNN`, all trained with a manual loop. |
| `src/train_digits.py` | Trains all three models on digits and writes `results/digits_results.json`. |
| `src/train_cifar10.py` | Trains the residual CNN on CIFAR-10 for 30 epochs with per-epoch checkpointing; writes `results/cifar10_results.json`. |
| `docs/notes_backprop.md` | The backpropagation and Adam derivations the NumPy implementation follows. |
| `tests/` | Unit tests including the numerical gradient check (3 tests). |

## 5. Interface Design

The system has no graphical interface; its interface is the command line plus the JSON result files.

### 5.1 Digits Training Run

Command: `.venv/bin/python -m src.train_digits`
The screen shows the split sizes, loss lines for the NumPy network at epochs 100, 200, and 300, then one summary line per model with its test accuracy and seconds taken. The full run takes about 40 seconds.

### 5.2 CIFAR-10 Training Run

Command: `.venv/bin/python -m src.train_cifar10`
The screen shows one line per epoch (training loss and test accuracy). A checkpoint is written after every epoch; re-running the command resumes from the last completed epoch. The full 30-epoch run takes roughly 4 to 6 hours on CPU.

### 5.3 Gradient Check and Unit Tests

Command: `.venv/bin/python -m unittest discover -s tests`
Runs 3 tests, including the numerical gradient check that verifies the NumPy backpropagation before any training result is trusted.

### 5.4 Outputs and Reports

- `results/digits_results.json`: per-model test accuracy, timing, and training history for the digits benchmark.
- `results/cifar10_results.json`: per-epoch history and final test accuracy for the CIFAR-10 run.
- Console output during training serves as the live progress report.

## Appendix A: Project Timeline and Measured Results

- **6 November 2026:** NumPy network, Adam, and gradient check built and verified; PyTorch MLP and residual CNN built; digits benchmark completed.
- **7–8 November 2026:** CIFAR-10 full run executed with per-epoch checkpointing (survived machine interruptions); completed all 30 epochs.

**Measured results (from the saved JSON files):**

| Benchmark | Model | Test accuracy |
|---|---|---|
| Digits | NumPy MLP (from scratch) | 96.9% |
| Digits | PyTorch MLP (custom) | 97.2% |
| Digits | Residual CNN (custom) | 99.7% |
| CIFAR-10 | Residual CNN (custom), 30 epochs | 75.8% final (best epoch 80.5%) |
