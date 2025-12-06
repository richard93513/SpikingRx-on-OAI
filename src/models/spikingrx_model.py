# src/models/spikingrx_model.py
"""
Lightweight SpikingRx for GTX950M
Conv-based SNN on GPU, ANN readout on CPU.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sew_block import SEWBlock
from .conv_block import ConvBlock
from .norm_layer import SpikeNorm
from .lif_neuron import LIF


# =============================
#  Stem Conv
# =============================
class StemConv(nn.Module):
    def __init__(self, in_ch, out_ch, beta=0.9, theta=0.5):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = SpikeNorm(out_ch)
        self.lif = LIF(beta=beta, theta=theta)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        outs = []
        for t in range(T):
            z = self.conv(x[:, t])   # GPU
            z = self.norm(z)
            outs.append(z)
        out = torch.stack(outs, dim=1)
        out = self.lif(out)
        return out


# =============================
#  ANN Readout 在 CPU
# =============================
class ReadoutANN(nn.Module):
    def __init__(self, in_ch, H=32, W=32, out_bits=14400):
        super().__init__()

        self.in_ch = in_ch
        self.H = H
        self.W = W
        self.out_bits = out_bits

        in_dim = in_ch * H * W
        hidden = max(256, in_dim // 10)

        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_bits)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x on CPU
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================
#     SpikingRx Model
# =============================
class SpikingRxModel(nn.Module):
    def __init__(
        self,
        in_ch=2,
        base_ch=12,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
        out_bits=14400,
        T=3,
        device_conv=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_fc=torch.device("cpu"),
    ):
        super().__init__()

        # Save configs
        self.T = T
        self.bits_per_symbol = bits_per_symbol
        self.llr_temperature = llr_temperature
        self.device_conv = device_conv
        self.device_fc = device_fc
        self.out_bits = out_bits

        # 1) Stem
        self.stem = StemConv(in_ch, base_ch, beta, theta)

        # 2) 6 × SEW blocks（縮小 channel）
        chs = [
            (base_ch, base_ch),
            (base_ch, base_ch * 2),
            (base_ch * 2, base_ch * 2),
            (base_ch * 2, base_ch * 3),
            (base_ch * 3, base_ch * 3),
            (base_ch * 3, base_ch * 3),
        ]
        self.stages = nn.ModuleList([SEWBlock(a, b) for a, b in chs])
        final_ch = chs[-1][1]

        # 3) ANN Readout（CPU）
        self.readout = ReadoutANN(final_ch, H=32, W=32, out_bits=out_bits)

        # ============================
        #  裝置分配：SNN → GPU，ANN → CPU
        # ============================
        self.stem.to(self.device_conv)
        self.stages.to(self.device_conv)
        self.readout.to(self.device_fc)

    # ====================================
    #            Forward Pass
    # ====================================
    def forward(self, x):
        # x: [B, T, C, H, W] on CPU

        # ---- Spiking 部分上 GPU ----
        x = x.to(self.device_conv)

        B, T, C, H, W = x.shape
        assert T == self.T, f"Input has T={T}, expected T={self.T}"

        # 1) Stem
        out = self.stem(x)

        # 2) 6 × SEW blocks
        spike_rates = []
        for stage in self.stages:
            out = stage(out)
            r = out.clamp(0, 1).mean(dim=(0, 2, 3, 4))  # [T]
            spike_rates.append(r)

        # 3) Temporal average
        rate = out.clamp(0, 1).mean(dim=1)  # [B, C, H, W]

        # ---- ANN Readout on CPU ----
        rate_cpu = rate.to(self.device_fc)
        logits = self.readout(rate_cpu) * self.llr_temperature

        aux = {
            "spike_rate_per_stage": torch.stack(spike_rates).detach().cpu(),
            "final_rate_mean": rate_cpu.mean().item(),
            "final_rate_std": rate_cpu.std().item(),
        }

        return logits, aux



