# src/models/spikingrx_model.py
"""
SpikingRx serial-pipeline model for OAI bundle training.

Input:
    x: [B, T, C, H, W]
       default now: [B, 1, 4, 14, 1272]

Output:
    pred: dict
        pred["llr"] : [B, out_bits]
        pred["eq"]  : [B, 2, H, W]
        pred["ch"]  : [B, 2, H, W]
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sew_block import SEWBlock
from .conv_block import ConvBlock
from .norm_layer import SpikeNorm
from .lif_neuron import LIF


class StemConv(nn.Module):
    def __init__(self, in_ch, out_ch, beta=0.9, theta=0.2, input_gain=5.0):
        super().__init__()
        self.conv = ConvBloc k(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = SpikeNorm(out_ch)
        self.lif = LIF(beta=beta, theta=theta)
        self.input_gain = input_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        return: [B, T, out_ch, H, W]
        """
        B, T, C, H, W = x.shape
        outs = []

        for t in range(T):
            z = self.conv(x[:, t])
            z = self.norm(z)
            z = z * self.input_gain
            outs.append(z)

        z_all = torch.stack(outs, dim=1)
        out_spike = self.lif(z_all)
        return out_spike


class DenseHead(nn.Module):
    """
    Dense 2-channel prediction head for:
      - channel estimate
      - equalized / compensated symbols
      - per-RE 2-bit LLR map
    """
    def __init__(self, in_ch: int, hidden_ch: int = 32, out_ch: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, padding=0, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, out_ch, H, W]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class EqFusionHead(nn.Module):
    """
    Equalization / compensation head that consumes:
      shared feature + predicted channel
    """
    def __init__(self, feat_ch: int, ch_pred_ch: int = 2, hidden_ch: int = 32, out_ch: int = 2):
        super().__init__()
        in_ch = feat_ch + ch_pred_ch
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, padding=0, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv3.bias)

    def forward(self, feat: torch.Tensor, ch_pred: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, ch_pred], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class LLRDenseHead(nn.Module):
    """
    Per-RE LLR head:
      (shared feature + ch_pred + eq_pred) -> [B, 2, H, W]
    Then gather only data REs and flatten to [B, G].
    """
    def __init__(self, feat_ch: int, aux_ch: int = 4, hidden_ch: int = 32):
        super().__init__()
        in_ch = feat_ch + aux_ch
        self.head = DenseHead(in_ch=in_ch, hidden_ch=hidden_ch, out_ch=2)

    def forward(self, feat: torch.Tensor, ch_pred: torch.Tensor, eq_pred: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, ch_pred, eq_pred], dim=1)
        return self.head(x)  # [B,2,H,W]


class SpikingRxModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 4,
        base_ch: int = 8,
        bits_per_symbol: int = 2,
        beta: float = 0.9,
        theta: float = 0.2,
        llr_temperature: float = 2.0,
        out_bits: int = 14400,
        T: int = 1,
        device_conv: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_fc: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        input_gain: float = 5.0,
    ):
        super().__init__()

        self.T = T
        self.bits_per_symbol = bits_per_symbol
        self.llr_temperature = llr_temperature
        self.device_conv = device_conv
        self.device_fc = device_fc
        self.out_bits = out_bits

        # -------------------------------------------------
        # shared encoder
        # -------------------------------------------------
        self.stem = StemConv(in_ch, base_ch, beta=beta, theta=theta, input_gain=input_gain)

        chs = [
            (base_ch, base_ch),         # 8 -> 8
            (base_ch, base_ch * 2),     # 8 -> 16
            (base_ch * 2, base_ch * 2), # 16 -> 16
            (base_ch * 2, base_ch * 3), # 16 -> 24
            (base_ch * 3, base_ch * 3), # 24 -> 24
            (base_ch * 3, base_ch * 3), # 24 -> 24
        ]
        self.stages = nn.ModuleList([
            SEWBlock(in_c, out_c, stride=1, beta=beta, theta=theta)
            for (in_c, out_c) in chs
        ])
        final_ch = chs[-1][1]

        # -------------------------------------------------
        # serial pipeline heads
        # -------------------------------------------------
        self.ch_head = DenseHead(
            in_ch=final_ch,
            hidden_ch=max(16, final_ch),
            out_ch=2,
        )

        self.eq_head = EqFusionHead(
            feat_ch=final_ch,
            ch_pred_ch=2,
            hidden_ch=max(16, final_ch),
            out_ch=2,
        )

        self.llr_head = LLRDenseHead(
            feat_ch=final_ch,
            aux_ch=4,
            hidden_ch=max(16, final_ch),
        )

        # -------------------------------------------------
        # device placement
        # -------------------------------------------------
        self.stem.to(self.device_conv)
        self.stages.to(self.device_conv)
        self.ch_head.to(self.device_conv)
        self.eq_head.to(self.device_conv)
        self.llr_head.to(self.device_conv)

    def _gather_llr_from_data_mask(
        self,
        llr_map: torch.Tensor,
        data_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        llr_map   : [B, 2, H, W]
        data_mask : [B, H, W]
        return    : [B, G] where G = (#data_re * 2)
        """
        B, C, H, W = llr_map.shape
        assert C == self.bits_per_symbol, (
            f"llr_map channel={C}, expected bits_per_symbol={self.bits_per_symbol}"
        )

        outs = []
        for b in range(B):
            pos = torch.nonzero(data_mask[b] > 0.5, as_tuple=False)  # [N,2]
            if pos.numel() == 0:
                raise RuntimeError("data_mask has no active RE")

            s_idx = pos[:, 0].long()
            w_idx = pos[:, 1].long()

            # [2, N] -> [N, 2] -> [2N]
            bits = llr_map[b, :, s_idx, w_idx].transpose(0, 1).contiguous().reshape(-1)

            if bits.numel() != self.out_bits:
                raise RuntimeError(
                    f"Gathered llr length mismatch: got {bits.numel()}, expected {self.out_bits}"
                )
            outs.append(bits)

        return torch.stack(outs, dim=0)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C, H, W]

        returns
        -------
        pred : dict
            pred["llr"] : [B, out_bits]
            pred["eq"]  : [B, 2, H, W]
            pred["ch"]  : [B, 2, H, W]

        aux : dict
            debug/stat info
        """
        x = x.to(self.device_conv)

        B, T, C, H, W = x.shape
        assert T == self.T, f"Input has T={T}, expected T={self.T}"

        # shared spiking encoder
        out = self.stem(x)  # [B,T,C,H,W]

        spike_rates = []
        for stage in self.stages:
            out = stage(out)
            r = out.clamp(0, 1).mean(dim=(0, 2, 3, 4))
            spike_rates.append(r)

        # time-mean shared feature
        feat = out.clamp(0, 1).mean(dim=1)  # [B,C,H,W]

        # serial pipeline
        ch_pred = self.ch_head(feat)                # [B,2,H,W]
        eq_pred = self.eq_head(feat, ch_pred)       # [B,2,H,W]
        llr_map = self.llr_head(feat, ch_pred, eq_pred)  # [B,2,H,W]

        # x channel:
        #   0 real
        #   1 imag
        #   2 dmrs_mask
        #   3 data_mask
        data_mask = x[:, 0, 3, :, :]  # [B,H,W]

        llr_pred = self._gather_llr_from_data_mask(llr_map, data_mask)
        llr_pred = llr_pred * self.llr_temperature

        pred = {
            "llr": llr_pred,
            "eq": eq_pred,
            "ch": ch_pred,
        }

        aux = {
            "spike_rate_per_stage": torch.stack(spike_rates).detach().cpu(),
            "final_rate_mean": feat.detach().mean().item(),
            "final_rate_std": feat.detach().std().item(),
            "ch_mean_abs": ch_pred.detach().abs().mean().item(),
            "eq_mean_abs": eq_pred.detach().abs().mean().item(),
            "llr_map_mean_abs": llr_map.detach().abs().mean().item(),
        }

        return pred, aux
