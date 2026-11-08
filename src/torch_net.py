"""The same ideas in PyTorch, written the way a research lab writes them:
custom nn.Module classes and a hand-written training loop.

No Trainer, no Lightning, no high-level wrapper: the loop below calls
zero_grad / forward / loss / backward / step explicitly, modifies nothing it
does not name, and logs loss and accuracy per epoch. TorchMLP mirrors the
NumPy MLP so the two implementations can be compared directly.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchMLP(nn.Module):
    def __init__(self, in_features: int = 64, hidden: int = 128,
                 n_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)
        # Explicit initialisation, not the default: He/Kaiming for the ReLU
        # layer, Xavier for the classifier head.
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class ResidualBlock(nn.Module):
    """A small convolutional residual block, written out explicitly."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)  # skip connection


class SmallResCNN(nn.Module):
    """CNN for 8x8 single-channel images (sklearn digits), with one
    custom residual block. Small enough to train on a laptop CPU."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.res = ResidualBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.res(self.stem(x))
        return self.head(self.pool(out).flatten(1))


def make_loader(x, y, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: str = "cpu") -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(dim=1).cpu()
        correct += int((pred == yb).sum())
        total += len(yb)
    return correct / total


def train_model(model: nn.Module, train_loader: DataLoader,
                test_loader: DataLoader, epochs: int = 25,
                lr: float = 1e-3, device: str = "cpu") -> list[dict]:
    """Hand-written training loop. Returns per-epoch history."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss) * len(xb)
        history.append({
            "epoch": epoch,
            "train_loss": round(running / len(train_loader.dataset), 4),
            "test_accuracy": round(evaluate(model, test_loader, device), 4),
        })
    return history
