# src/models/lif_neuron.py
# ======================================================
#  Lightweight SpikingRx – LIF Neuron (論文正統版本)
# ======================================================

import torch
import torch.nn as nn

# ------------------------------------------------------
#  Triangular Surrogate Gradient
#  σ'(x) = max(0, 1 - |x|/a) / a
# ------------------------------------------------------

class TriangularSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a=1.0):
        ctx.save_for_backward(x)
        ctx.a = a
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        a = ctx.a

        mask = (x.abs() < a).to(x.dtype)
        grad = grad_output * (1 - x.abs() / a) * mask / a
        return grad, None


def spike_fn(x, a=1.0):
    return TriangularSurrogate.apply(x, a)


# ------------------------------------------------------
#  LIF Neuron – 論文正統實作
# ------------------------------------------------------

class LIF(nn.Module):
    """
    U[t] = βU[t−1] + (1−β)I[t]
    S[t] = H(U[t] − θ)
    U[t] = U[t] − S[t]θ
    """

    def __init__(self, beta=0.9, theta=1.0):
        super().__init__()

        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("theta", torch.tensor(float(theta)))

    def forward(self, I):
        # 支援 2 種維度：
        # [B,T,C,H,W] → 卷積層
        # [B,T,D]     → 全連接層
        if I.dim() == 5:
            B, T, C, H, W = I.shape
            U = I.new_zeros((B, C, H, W))
            outs = []

            for t in range(T):
                It = I[:, t]
                U = self.beta * U + (1 - self.beta) * It
                S = spike_fn(U - self.theta)
                U = U - S * self.theta
                outs.append(S)

            return torch.stack(outs, dim=1)

        elif I.dim() == 3:
            B, T, D = I.shape
            U = I.new_zeros((B, D))
            outs = []

            for t in range(T):
                It = I[:, t]
                U = self.beta * U + (1 - self.beta) * It
                S = spike_fn(U - self.theta)
                U = U - S * self.theta
                outs.append(S)

            return torch.stack(outs, dim=1)

        else:
            raise ValueError("LIF input must be [B,T,C,H,W] or [B,T,D]")

# --------------------------------------------------------
