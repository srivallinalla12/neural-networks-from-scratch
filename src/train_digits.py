"""Experiment: three implementations, one dataset, one split.

Dataset: sklearn digits (1,797 handwritten 8x8 digits, 10 classes). Small,
real, and trains in under a minute on a laptop CPU, which makes it a good
honest testbed: every number in results/ is reproducible from this file.

Models:
  1. NumpyMLP    -- from scratch in NumPy, manual backprop + manual Adam
  2. TorchMLP    -- the same architecture in PyTorch, manual training loop
  3. SmallResCNN -- PyTorch CNN with a custom residual block

Run: .venv/bin/python -m src.train_digits
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .numpy_net import Adam, NumpyMLP
from .torch_net import SmallResCNN, TorchMLP, make_loader, train_model

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    torch.manual_seed(0)
    digits = load_digits()
    x, y = digits.data / 16.0, digits.target  # pixel values are 0..16
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y)
    print(f"train {len(x_tr)}  test {len(x_te)}")

    results = {"dataset": "sklearn digits, stratified 80/20 split, seed 0",
               "models": {}}

    # 1. NumPy from scratch, full-batch training.
    t0 = time.time()
    net = NumpyMLP(in_features=64, hidden=128, n_classes=10, seed=0)
    opt = Adam(lr=0.01)
    for epoch in range(300):
        loss = net.train_step(x_tr, y_tr, opt)
        if (epoch + 1) % 100 == 0:
            print(f"numpy epoch {epoch + 1}: loss {loss:.4f}")
    acc = net.accuracy(x_te, y_te)
    results["models"]["numpy_mlp_scratch"] = {
        "test_accuracy": round(acc, 4), "seconds": round(time.time() - t0, 1),
        "notes": "Manual forward/backward, He init, hand-written Adam, full batch, 300 epochs"}
    print("numpy_mlp_scratch", results["models"]["numpy_mlp_scratch"])

    # 2 and 3. PyTorch models share loaders and the same manual loop.
    train_loader = make_loader(x_tr, y_tr)
    test_loader = make_loader(x_te, y_te, shuffle=False)
    img_tr = x_tr.reshape(-1, 1, 8, 8)
    img_te = x_te.reshape(-1, 1, 8, 8)

    for name, model, loader_tr, loader_te, epochs in [
        ("torch_mlp", TorchMLP(), train_loader, test_loader, 25),
        ("torch_rescnn", SmallResCNN(),
         make_loader(img_tr, y_tr), make_loader(img_te, y_te, shuffle=False), 25),
    ]:
        t0 = time.time()
        hist = train_model(model, loader_tr, loader_te, epochs=epochs)
        results["models"][name] = {
            "test_accuracy": hist[-1]["test_accuracy"],
            "final_train_loss": hist[-1]["train_loss"],
            "seconds": round(time.time() - t0, 1),
            "history": hist,
        }
        print(name, results["models"][name]["test_accuracy"],
              f"({results['models'][name]['seconds']}s)")

    out = ROOT / "results" / "digits_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
