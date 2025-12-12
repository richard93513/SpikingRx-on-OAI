# src/models/spikingrx_model.py
"""
Lightweight SpikingRx for GTX950M
Spiking conv trunk on GPU, ANN readout on CPU.

輸入：
    x: [B, T, C, H, W]  (例如 B=1, T=3, C=2, H=W=32)

輸出：
    logits: [B, out_bits]  (對應 14400 LLR)
    aux:
        - spike_rate_per_stage: [num_stages, T]
        - final_rate_mean, final_rate_std
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
#  Stem Conv (第一層 spiking conv)
# =============================
class StemConv(nn.Module):
    def __init__(self, in_ch, out_ch, beta=0.9, theta=0.5):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = SpikeNorm(out_ch)
        self.lif = LIF(beta=beta, theta=theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        return: [B, T, out_ch, H, W]
        """
        B, T, C, H, W = x.shape
        outs = []

        # 先對每個時間步做 Conv + Norm
        for t in range(T):
            z = self.conv(x[:, t])   # [B,out_ch,H,W]
            z = self.norm(z)         # SpikeNorm (單一時間步)
            outs.append(z)

        z_all = torch.stack(outs, dim=1)  # [B,T,out_ch,H,W]

        # 再交給 LIF 做時序整合 + spike
        out_spike = self.lif(z_all)       # [B,T,out_ch,H,W]
        return out_spike


# =============================
#  ANN Readout 在 CPU，輸出 LLR
# =============================
class ReadoutANN(nn.Module):
    def __init__(self, in_ch, H=32, W=32, out_bits=14400):
        super().__init__()
        self.in_ch = in_ch
        self.H = H
        self.W = W
        self.out_bits = out_bits

        in_dim = in_ch * H * W
        hidden = max(256, in_dim // 10)  # 輕量版，但仍保留足夠 capacity

        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_bits)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] on CPU
        x = x.view(x.size(0), -1)   # [B, C*H*W]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================
#     SpikingRx Model 本體
# =============================
class SpikingRxModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 2,
        base_ch: int = 16,          # 輕量版建議 16
        bits_per_symbol: int = 2,
        beta: float = 0.9,
        theta: float = 0.5,
        llr_temperature: float = 1.0,
        out_bits: int = 14400,
        T: int = 3,
        device_conv: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_fc: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        # Config 保存
        self.T = T
        self.bits_per_symbol = bits_per_symbol
        self.llr_temperature = llr_temperature
        self.device_conv = device_conv
        self.device_fc = device_fc
        self.out_bits = out_bits

        # ---------- 1) Stem ----------
        self.stem = StemConv(in_ch, base_ch, beta=beta, theta=theta)

        # ---------- 2) SEW stages (輕量版通道設計) ----------
        # 6 個 block：channel 緩慢增加，避免 950M 爆掉
        chs = [
            (base_ch, base_ch),         # 16 → 16
            (base_ch, base_ch * 2),     # 16 → 32
            (base_ch * 2, base_ch * 2), # 32 → 32
            (base_ch * 2, base_ch * 3), # 32 → 48
            (base_ch * 3, base_ch * 3), # 48 → 48
            (base_ch * 3, base_ch * 3), # 48 → 48
        ]
        self.stages = nn.ModuleList([
            SEWBlock(in_c, out_c, stride=1, beta=beta, theta=theta)
            for (in_c, out_c) in chs
        ])
        final_ch = chs[-1][1]

        # ---------- 3) ANN Readout（CPU） ----------
        self.readout = ReadoutANN(final_ch, H=32, W=32, out_bits=out_bits)

        # ============================
        #  裝置分配：SNN trunk → GPU，ANN head → CPU
        # ============================
        self.stem.to(self.device_conv)
        self.stages.to(self.device_conv)
        self.readout.to(self.device_fc)

    # ====================================
    #            Forward Pass
    # ====================================
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        x: [B, T, C, H, W] on CPU
        return:
            logits: [B, out_bits]
            aux: dict (監控 spike 行為)
        """

        # ---- Spiking conv trunk on GPU ----
        x = x.to(self.device_conv)

        B, T, C, H, W = x.shape
        assert T == self.T, f"Input has T={T}, expected T={self.T}"

        # 1) Stem
        out = self.stem(x)  # [B,T,base_ch,H,W]

        # 2) 6 × SEW blocks
        spike_rates = []
        for stage in self.stages:
            out = stage(out)   # [B,T,C',H,W]
            # 監控每個 stage 的 spike rate（在所有空間 + batch 上平均）
            # out ∈ [0,1]（spike train），mean over B,C,H,W → [T]
            r = out.clamp(0, 1).mean(dim=(0, 2, 3, 4))  # [T]
            spike_rates.append(r)

        # 3) Temporal average（T 維度平均）→ rate coding
        rate = out.clamp(0, 1).mean(dim=1)  # [B,C,H,W]

        # ---- ANN Readout on CPU ----
        rate_cpu = rate.to(self.device_fc)       # 移到 CPU
        logits = self.readout(rate_cpu)          # [B,out_bits]
        logits = logits * self.llr_temperature   # 保留 temperature 介面

        # ---- Aux info（可視覺化用）----
        aux = {
            "spike_rate_per_stage": torch.stack(spike_rates).detach().cpu(),  # [num_stages,T]
            "final_rate_mean": rate_cpu.mean().item(),
            "final_rate_std": rate_cpu.std().item(),
        }

        return logits, aux


