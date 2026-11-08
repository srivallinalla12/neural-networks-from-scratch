"""CIFAR-10: the Project 2 main experiment.

Dataset: 60,000 small colour images (32x32), 10 classes
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
50,000 for training, 10,000 for testing. Downloaded automatically by
torchvision on first run into ./data (about 170 MB).

Model: a CIFAR-sized version of the same residual CNN idea used on digits:
3-channel stem, two custom ResidualBlocks with more channels, global
average pooling, linear head. Trained with the SAME hand-written loop from
torch_net.train_model, plus simple augmentation (random crop + flip).

Full run (local machine or a free cloud GPU, about 30 epochs):
  .venv/bin/python -m src.train_cifar10 --epochs 30

Smoke run (verifies everything end to end on a small slice, minutes on CPU):
  .venv/bin/python -m src.train_cifar10 --smoke
Smoke numbers are pipeline checks only and are labelled as such in the
output file. Only full runs may be quoted as results.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .torch_net import ResidualBlock, evaluate

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


class CifarResCNN(nn.Module):
    """Residual CNN for 32x32x3 images, built from the repo's own block."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.res1 = ResidualBlock(32)
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.res2 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.res1(self.stem(x))
        out = self.res2(self.down(out))
        return self.head(self.pool(out).flatten(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="4,000 train / 1,000 test, 2 epochs, CPU check only")
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(args.data, train=True, download=True,
                                transform=train_tf)
    test_ds = datasets.CIFAR10(args.data, train=False, download=True,
                               transform=test_tf)
    epochs = args.epochs
    if args.smoke:
        train_ds = Subset(train_ds, range(4000))
        test_ds = Subset(test_ds, range(1000))
        epochs = 2
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    print(f"device {device} | train {len(train_ds)} | test {len(test_ds)} "
          f"| epochs {epochs} | smoke {args.smoke}")

    model = CifarResCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    start_epoch = 0
    ckpt_path = ROOT / "results" / "cifar10_checkpoint.pt"
    if not args.smoke and ckpt_path.exists():
        # Resume after an interruption (this VM is replaced periodically,
        # which kills plain background processes). The checkpoint is
        # written after every epoch, so at most one epoch is ever lost.
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt["history"]
        start_epoch = int(ckpt["epoch"])
        print(f"resumed from checkpoint at epoch {start_epoch}")
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(xb)
        acc = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": round(running / len(train_ds), 4),
               "test_accuracy": round(acc, 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(row, flush=True)
        if not args.smoke:
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "epoch": epoch, "history": history}, ckpt_path)

    result = {"dataset": "CIFAR-10 (torchvision), standard train/test split",
              "smoke_run": bool(args.smoke),
              "model": "CifarResCNN: stem + 2 custom ResidualBlocks, Adam lr 1e-3, crop+flip augmentation",
              "final_test_accuracy": history[-1]["test_accuracy"],
              "history": history}
    name = "cifar10_smoke.json" if args.smoke else "cifar10_results.json"
    out = ROOT / "results" / name
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
