# src/models/conv_block.py

# --------------------------------------------------------
# 模組位置對應：SpikingRx → SEW-ResNet block → Conv 卷積層
# --------------------------------------------------------
# 這個檔案 conv_block.py 對應 SpikingRx 論文架構中的「卷積前端」：
# 在每個 SEW-ResNet block 的起始位置：
#    Conv → Norm → LIF(spike) → Shortcut
# ConvBlock 的功能是萃取輸入特徵，並提供連續值給後續的 Norm 層。
# --------------------------------------------------------

import torch
import torch.nn as nn

# ========================================================
# 一、卷積模組 (ConvBlock)
# ========================================================
# 功能：
#   - 負責將輸入 feature map 做空間上的特徵提取。
#   - 輸出仍為連續值（非 spike），會交由下一層 SpikeNorm + LIF 處理。
#   - 權重使用 Kaiming Normal 初始化，以適配 ReLU-like 的 spiking 動態。
# ========================================================

class ConvBlock(nn.Module):
    """
    SpikingRx 卷積模組：
      對應 SEW-ResNet block 的第一層卷積 (Convolution layer)
      ConvBlock = Conv2d + Kaiming 初始化
      不包含 Norm 或啟動函數（Activation），
      因為這部分由後續的 SpikeNorm + LIF 負責。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bias=True):
        super(ConvBlock, self).__init__()  # 初始化父類別 (nn.Module)

        # ----------------------------------------------------
        # 定義卷積層：
        #   - in_channels:  輸入通道數
        #   - out_channels: 輸出通道數（feature map 數量）
        #   - kernel_size:  卷積核大小 (3x3)
        #   - stride:       步幅（預設1）
        #   - padding:      補零，確保輸出尺寸不變
        #   - bias:         是否使用偏置項
        # ----------------------------------------------------
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias
        )

        # ----------------------------------------------------
        # 二、權重初始化 (Kaiming Normal)
        # ----------------------------------------------------
        # Kaiming Normal 是為 ReLU 類激活函數設計的初始化方法。
        # 在 SpikingRx 中，LIF 層的發火行為與 ReLU 類似（非線性截斷），
        # 因此使用相同原則可穩定梯度分佈。
        #   - mode='fan_out'：使輸出通道方差一致
        #   - nonlinearity='relu'：對應 spike 的啟動特性
        # ----------------------------------------------------
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        # 若啟用 bias，將偏置初始化為 0
        if use_bias:
            nn.init.constant_(self.conv.bias, 0.0)

    # ----------------------------------------------------
    # 三、Forward 運算流程
    # ----------------------------------------------------
    # 輸入  : [B, C_in, H, W]
    # 輸出  : [B, C_out, H, W]
    # 說明  :
    #   ConvBlock 不包含 spike 運算，
    #   只負責在空間維度上卷積輸入特徵，
    #   並將輸出交由下一層 Norm + LIF 處理。
    # ----------------------------------------------------
    def forward(self, x):
        out = self.conv(x)  # 卷積運算
        return out          # 回傳連續特徵圖 (非二值化)

# --------------------------------------------------------
# 模組小結：
# - ConvBlock = 單純卷積 + Kaiming 初始化
# - 不包含 LIF 或正規化，後續由 SpikeNorm + LIF 處理
# - 對應 SEW-ResNet block 的第一步：「特徵提取」
# --------------------------------------------------------


