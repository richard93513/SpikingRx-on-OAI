# src/models/conv_block.py
# ======================================================
#  Lightweight SpikingRx – Conv Block
# （論文精神版：乾淨、無 BN、Kaiming 初始化）
# ======================================================

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    SpikingRx 前端卷積模組（不含 BN、不含 activation）

    功能：
    - 單純進行 2D 卷積（spiking 動態交給 SpikeNorm + LIF）
    - 採用 Kaiming Normal 初始化（適合 ReLU/LIF 類非線性）
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # --------------------------------------------
        # Kaiming Normal 初始化（論文一致的初始化方式）
        # --------------------------------------------
        nn.init.kaiming_normal_(
            self.conv.weight,
            mode='fan_out',
            nonlinearity='relu'  # spike 函數類似 ReLU
        )
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        # x : [B, C, H, W]
        return self.conv(x)

