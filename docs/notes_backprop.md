# Notes: the derivations behind numpy_net.py

The mathematics behind `numpy_net.py`, derived in full.

## Softmax + cross-entropy gradient

For logits z, probabilities p = softmax(z), and one-hot target y, the
cross-entropy loss gradient simplifies to:

    dL/dz = (p - y) / N

That is why `SoftmaxCrossEntropy.backward` is three lines. The softmax
Jacobian and the log derivative cancel, leaving prediction minus target.

## Linear layer gradients

For out = x @ W + b with upstream gradient dout:

    dW = x^T @ dout
    db = sum of dout over the batch
    dx = dout @ W^T

## ReLU gradient

1 where the input was positive, 0 elsewhere. Implemented as a stored mask.

## Adam bias correction

Adam keeps exponential averages of the gradient (m) and squared gradient
(v). Both start at zero, so early estimates are biased toward zero. Dividing
by (1 - beta^t) corrects that. Skipping the correction makes early training
steps too small. `Adam.step` in numpy_net.py implements both averages and
the correction explicitly.

## He initialisation

Weights drawn with variance 2 / fan_in keep activation variance roughly
constant through ReLU layers (ReLU zeroes about half its inputs). Xavier
init (variance 1 / fan_in style scaling) suits tanh/sigmoid and the final
classifier layer. Using the wrong one is a classic cause of a network that
silently learns nothing.

## Gradient checking

Compare analytic gradients against (f(x+e) - f(x-e)) / 2e for a few weights.
Agreement to about 1e-5 means the backward pass is correct. This is the
fastest way to debug a hand-written network, and the test suite does it for
the Linear + softmax-CE path.
