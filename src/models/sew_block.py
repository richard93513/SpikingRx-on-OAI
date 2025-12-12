# src/models/sew_block.py
# ======================================================
#  Lightweight SpikingRx – SEW Block (論文精神 + 輕量版)
# ======================================================

import torch
import torch.nn as nn

from .conv_block import ConvBlock
from .norm_layer import SpikeNorm
from .lif_neuron import LIF


class SEWBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) Block

    結構：
      main path: Conv → SpikeNorm → LIF
      shortcut:  Identity 或 1×1 Conv
      output:    SEW 加法（spike-aware residual）

    功能：
      - 保留殘差結構但避免 spike 堆疊造成放電不穩定
      - 提升深層 SNN 訓練穩定度
      - 提升 LLR quality（論文中最重要的貢獻之一）
    """

    def __init__(self, in_channels, out_channels, stride=1, beta=0.9, theta=0.5):
        super().__init__()

        # 主支路
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.norm = SpikeNorm(out_channels)
        self.lif = LIF(beta=beta, theta=theta)

        # 捷徑分支
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        return: [B, T, C_out, H, W]
        """
        B, T, C, H, W = x.shape
        outputs = []

        for t in range(T):

            # 取單一時間步
            xt = x[:, t]  # [B,C,H,W]

            # ---- Main path ----
            m = self.conv(xt)        # Conv
            m = self.norm(m)         # SpikeNorm
            m = self.lif(m.unsqueeze(1))[:, 0]  # LIF → [B,C,H,W]

            # ---- Shortcut path ----
            s = self.shortcut(xt)

            # ---- SEW spike-aware add ----
            # 論文精神：避免 m+s 導致 spike 爆炸
            # 使用 0.5*(m + s) 當作輕量化 SEW gating（小模型穩定最佳）
            y = 0.5 * (m + s)

            outputs.append(y)

        return torch.stack(outputs, dim=1)

