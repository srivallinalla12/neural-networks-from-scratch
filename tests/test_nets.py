import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.numpy_net import Linear, NumpyMLP, SoftmaxCrossEntropy


class TestGradientCheck(unittest.TestCase):
    """The test a research lab cares about: do the hand-written gradients
    match numerical gradients? If backprop is wrong, this fails."""

    def test_linear_plus_ce_gradients(self):
        rng = np.random.default_rng(1)
        layer = Linear(4, 3, rng)
        loss_fn = SoftmaxCrossEntropy()
        x = rng.normal(size=(5, 4))
        y = np.array([0, 1, 2, 1, 0])

        loss_fn.forward(layer.forward(x), y)
        layer.backward(loss_fn.backward())
        analytic = layer.dW.copy()

        eps = 1e-5
        numeric = np.zeros_like(layer.W)
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                orig = layer.W[i, j]
                layer.W[i, j] = orig + eps
                lp = loss_fn.forward(layer.forward(x), y)
                layer.W[i, j] = orig - eps
                lm = loss_fn.forward(layer.forward(x), y)
                layer.W[i, j] = orig
                numeric[i, j] = (lp - lm) / (2 * eps)
        self.assertTrue(np.allclose(analytic, numeric, atol=1e-5),
                        f"max grad error {np.abs(analytic - numeric).max()}")


class TestNumpyTraining(unittest.TestCase):
    def test_loss_decreases_and_learns_toy_problem(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 4))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        from src.numpy_net import Adam
        net = NumpyMLP(4, 32, 2, seed=0)
        opt = Adam(lr=0.02)
        first = net.train_step(x, y, opt)
        for _ in range(200):
            last = net.train_step(x, y, opt)
        self.assertLess(last, first)
        self.assertGreater(net.accuracy(x, y), 0.9)


class TestTorchNets(unittest.TestCase):
    def test_forward_shapes_and_one_step(self):
        import torch
        from src.torch_net import SmallResCNN, TorchMLP
        mlp = TorchMLP()
        self.assertEqual(mlp(torch.randn(7, 64)).shape, (7, 10))
        cnn = SmallResCNN()
        self.assertEqual(cnn(torch.randn(7, 1, 8, 8)).shape, (7, 10))
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        loss = torch.nn.CrossEntropyLoss()(
            mlp(torch.randn(16, 64)), torch.randint(0, 10, (16,)))
        loss.backward()
        opt.step()
        self.assertTrue(np.isfinite(float(loss)))


if __name__ == "__main__":
    unittest.main()
