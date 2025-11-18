# src/models/spikingrx_model.py
"""
SpikingRx 論文級架構
StemConv (Conv+Norm+LIF) + 6×SEW Blocks + ANN Readout (LLR)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sew_block import SEWBlock
from .conv_block import ConvBlock
from .norm_layer import SpikeNorm
from .lif_neuron import LIF


# --- Stem Conv 模組 ---
class StemConv(nn.Module):
    """最前端預處理層：Conv → Norm → LIF"""
    def __init__(self, in_ch: int, out_ch: int, beta=0.9, theta=0.5):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = SpikeNorm(out_ch)
        self.lif = LIF(beta=beta, theta=theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        outs = []
        for t in range(T):
            out_t = self.conv(x[:, t])
            out_t = self.norm(out_t)
            outs.append(out_t)
        out = torch.stack(outs, dim=1)
        out = self.lif(out)
        return out


# --- Readout 層 ---
class ReadoutANN(nn.Module):
    """逐像素 LLR 讀出"""
    def __init__(self, in_ch: int, bits_per_symbol: int = 2):
        super().__init__()
        mid = max(8, in_ch // 2)
        self.fc1 = nn.Conv2d(in_ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, bits_per_symbol, 1)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        return self.fc2(x)


# --- SpikingRx 主模型 ---
class SpikingRxModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 2,
        base_ch: int = 16,
        bits_per_symbol: int = 2,
        beta: float = 0.9,
        theta: float = 0.5,
        llr_temperature: float = 1.0
    ):
        super().__init__()
        self.bits_per_symbol = bits_per_symbol
        self.llr_temperature = llr_temperature

        # StemConv
        self.stem = StemConv(in_ch, base_ch, beta, theta)

        # 六層 SEW blocks（論文設計）
        chs = [
            (base_ch, base_ch),
            (base_ch, base_ch * 2),
            (base_ch * 2, base_ch * 2),
            (base_ch * 2, base_ch * 4),
            (base_ch * 4, base_ch * 4),
            (base_ch * 4, base_ch * 4),
        ]
        self.stages = nn.ModuleList(
            [SEWBlock(c_in, c_out) for (c_in, c_out) in chs]
        )

        self.readout = ReadoutANN(chs[-1][1], bits_per_symbol)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T, _, H, W = x.shape
        out = self.stem(x)
        spike_rates = []

        for stage in self.stages:
            out = stage(out)
            r_t = out.clamp(0, 1).mean(dim=(0, 2, 3, 4))
            spike_rates.append(r_t)

        rate = out.clamp(0, 1).mean(dim=1)       # 時間聚合
        logits = self.readout(rate) * self.llr_temperature
        llr = logits.permute(0, 2, 3, 1).contiguous()

        aux = {
            "spike_rate_per_stage": torch.stack(spike_rates, dim=0),
            "final_rate_mean": rate.mean().detach(),
            "final_rate_std": rate.std().detach(),
        }
        return llr, aux


