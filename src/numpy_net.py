"""A small neural network implemented from scratch in NumPy.

No autograd, no framework layers: every forward and backward pass below is
written out by hand, so the code itself demonstrates backpropagation, weight
initialisation, and Adam rather than only calling a high-level API.

Components:
  Linear      -- fully connected layer, He initialisation, manual gradients
  ReLU        -- activation with manual gradient (mask)
  SoftmaxCE   -- fused softmax + cross-entropy, manual gradient
  Adam        -- optimiser implemented from the paper's update rules
  NumpyMLP    -- Linear -> ReLU -> Linear classifier built from the above
"""
from __future__ import annotations

import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int,
                 rng: np.random.Generator):
        # He initialisation: keeps activation variance stable through ReLU
        # layers, which prevents vanishing gradients at depth.
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0, scale, size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._mask is not None
        return dout * self._mask


class SoftmaxCrossEntropy:
    """Fused softmax + cross-entropy.

    The gradient of this pair simplifies to (probs - one_hot) / N, which is
    derived in docs/notes_backprop.md and checked numerically in the tests.
    """

    def __init__(self):
        self._probs: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)  # stability
        exp = np.exp(shifted)
        self._probs = exp / exp.sum(axis=1, keepdims=True)
        self._y = y
        n = len(y)
        correct = self._probs[np.arange(n), y]
        return float(-np.log(correct + 1e-12).mean())

    def backward(self) -> np.ndarray:
        assert self._probs is not None and self._y is not None
        n = len(self._y)
        d = self._probs.copy()
        d[np.arange(n), self._y] -= 1.0
        return d / n


class Adam:
    """Adam optimiser (Kingma & Ba, 2015), implemented manually."""

    def __init__(self, lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self, params_and_grads) -> None:
        self.t += 1
        for p, g in params_and_grads:
            k = id(p)
            if k not in self._m:
                self._m[k] = np.zeros_like(p)
                self._v[k] = np.zeros_like(p)
            self._m[k] = self.b1 * self._m[k] + (1 - self.b1) * g
            self._v[k] = self.b2 * self._v[k] + (1 - self.b2) * (g ** 2)
            m_hat = self._m[k] / (1 - self.b1 ** self.t)   # bias correction
            v_hat = self._v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class NumpyMLP:
    def __init__(self, in_features: int, hidden: int, n_classes: int,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.fc1 = Linear(in_features, hidden, rng)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, n_classes, rng)
        self.loss_fn = SoftmaxCrossEntropy()

    def _logits(self, x: np.ndarray) -> np.ndarray:
        return self.fc2.forward(self.relu.forward(self.fc1.forward(x)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.loss_fn.forward(self._logits(x), y)

    def train_step(self, x: np.ndarray, y: np.ndarray,
                   opt: Adam) -> float:
        loss = self.loss(x, y)
        d = self.loss_fn.backward()
        d = self.fc2.backward(d)
        d = self.relu.backward(d)
        self.fc1.backward(d)
        opt.step(self.fc1.params_and_grads() + self.fc2.params_and_grads())
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._logits(x).argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())
