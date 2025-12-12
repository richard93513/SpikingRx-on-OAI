# src/models/norm_layer.py
# ======================================================
#  Lightweight SpikingRx – SpikeNorm (論文正統版本)
# ======================================================

import torch
import torch.nn as nn


class SpikeNorm(nn.Module):
    """
    SpikeNorm (論文精神版本)

    - 不使用 BatchNorm2d 的 running mean/var
    - 每個時間步 T 各自做通道層級標準化
    - 不跨時間步混合統計 → 保留時序稀疏性
    - 每個 channel 在 H×W 上做 μ/σ 標準化
    """

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        # ------------------------------------------------
        # case 1: x = [B, C, H, W]
        # ------------------------------------------------
        if x.dim() == 4:
            B, C, H, W = x.shape

            # 計算各通道 μ, σ
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)

            out = (x - mean) / torch.sqrt(var + self.eps)

            if self.gamma is not None:
                out = out * self.gamma.view(1, C, 1, 1)
                out = out + self.beta.view(1, C, 1, 1)

            return out

        # ------------------------------------------------
        # case 2: x = [B, T, C, H, W]
        # ------------------------------------------------
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            outs = []

            for t in range(T):
                xt = x[:, t]  # [B,C,H,W]

                mean = xt.mean(dim=[0, 2, 3], keepdim=True)
                var = xt.var(dim=[0, 2, 3], unbiased=False, keepdim=True)

                yt = (xt - mean) / torch.sqrt(var + self.eps)

                if self.gamma is not None:
                    yt = yt * self.gamma.view(1, C, 1, 1)
                    yt = yt + self.beta.view(1, C, 1, 1)

                outs.append(yt)

            return torch.stack(outs, dim=1)

        else:
            raise ValueError("SpikeNorm input must be 4D or 5D tensor")


