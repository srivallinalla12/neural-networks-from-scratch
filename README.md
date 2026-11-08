# Neural Networks From Scratch: NumPy and PyTorch

Author: Srivalli Nalla, MS Computer Science, George Mason University
Purpose: Deep, hands-on practice with deep learning fundamentals: strong Python, PyTorch, and the ability to train and modify custom architectures, all built and verified from first principles rather than called from a high-level API.

## The point in one line

Anyone can call a high-level API. This repo builds the same classifier three ways, including the version with **no framework at all**, where forward passes, backpropagation, and the Adam optimiser are written out by hand and then **proved correct with a numerical gradient check**.

## What is inside

- `src/numpy_net.py` A neural network from scratch in NumPy: Linear layer with He initialisation, ReLU, fused softmax cross-entropy, Adam implemented from the paper, manual forward and backward passes throughout.
- `src/torch_net.py` The same ideas in PyTorch, written the way a lab writes them: a custom MLP with explicit initialisation, a CNN with a hand-written residual block, and a manual training loop (zero_grad, forward, loss, backward, step). No Trainer or other high-level wrapper.
- `src/train_digits.py` One experiment, three models, one fixed split, so the results are comparable.
- `tests/test_nets.py` Includes the gradient check: analytic gradients from the hand-written backprop are compared against numerical gradients. If the maths were wrong, this test would fail. It passes.
- `docs/notes_backprop.md` The derivations behind the code, written out in full.

## Results (real runs, reproducible, CPU)

Dataset: sklearn digits, 1,797 handwritten 8x8 digits, stratified 80/20 split, seed 0. Raw output: `results/digits_results.json`.

| Model | Test accuracy | What it proves |
|---|---|---|
| NumPy MLP, from scratch | 96.9% | Backprop, initialisation, and Adam understood, not just used |
| PyTorch MLP, manual loop | 97.2% | Idiomatic PyTorch, custom module, explicit init |
| PyTorch CNN, custom residual block | 99.7% | Custom architecture design and modification |

These are honest starter results on a small benchmark.

## CIFAR-10 full run (completed November 2026)

Same custom residual CNN, trained from scratch on the full CIFAR-10 training set (50,000 images), 30 epochs, CPU, with per-epoch checkpointing. Raw output: `results/cifar10_results.json`.

| Result | Value |
|---|---|
| Final test accuracy (epoch 30) | **75.82%** |
| Best epoch | 23, at 80.53% |
| Late-epoch range (27 to 30) | 75.6% to 76.7% |

Training used per-epoch checkpoints, so an interrupted run resumes where it stopped. The accuracy curve is noisy epoch to epoch, which is normal for this setup (small network, CPU, basic augmentation: crop and flip only).

## Run it

```bash
cd neural-networks-from-scratch
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # numpy, scikit-learn, torch (CPU)
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.train_digits
```

## Skills demonstrated

Custom neural network architectures from scratch; training loops; PyTorch; backpropagation verified by gradient checking; weight initialisation (He, Xavier); optimisers (Adam, implemented manually); BatchNorm, dropout, residual connections; model evaluation on a held-out split.
