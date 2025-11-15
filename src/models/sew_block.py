# src/models/sew_block.py

import torch
import torch.nn as nn

from .conv_block import ConvBlock
from .norm_layer import SpikeNorm
from .lif_neuron import LIF


class SEWBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) Block 模組

    主支路 (main path):
        X[t] → Conv → Norm → LIF → main_out[t]

    捷徑支路 (shortcut):
        X[t] → (Identity 或 1×1 Conv) → shortcut_out[t]

    最終輸出:
        Y[t] = main_out[t] + shortcut_out[t]

    輸入:  [B, T, C_in, H, W]
    輸出:  [B, T, C_out, H, W]
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(SEWBlock, self).__init__()

        # 主支路
        self.conv = ConvBlock(in_channels, out_channels, stride=stride)
        self.norm = SpikeNorm(out_channels)
        self.lif = LIF(theta=0.2)  # 使用 [B,T,C,H,W] 格式

        # 捷徑分支：若維度不同需用 1×1 Conv 對齊
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
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

            xt = x[:, t]                          # [B,C,H,W]
            main = self.conv(xt)                  # Conv
            main = self.norm(main)                # Norm

            # LIF 需要 5 維，因此加回 T=1 維
            main = self.lif(main.unsqueeze(1))    # [B,1,C,H,W]
            main = main[:, 0]                     # 移除時間維度 → [B,C,H,W]

            sc = self.shortcut(xt)                # Shortcut

            outputs.append(main + sc)

        return torch.stack(outputs, dim=1)        # [B,T,C_out,H,W]

