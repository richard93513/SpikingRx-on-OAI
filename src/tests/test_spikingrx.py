# src/tests/test_spikingrx.py
# -*- coding: utf-8 -*-

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 讓 Python 找到 src/models
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.spikingrx_model import SpikingRxModel  # noqa: E402


def describe_tensor(name: str, x: torch.Tensor):
    """印出張量的 shape / mean / std。"""
    print(f"{name}: shape={tuple(x.shape)}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")


def build_dummy_ofdm_input(
    B: int = 2, T: int = 5, H: int = 32, W: int = 32, device="cpu"
) -> torch.Tensor:
    """
    建一個假的 OFDM 輸入：
      - C=2: I/Q
      - shape: [B, T, C, H, W]
    這裡先用 N(0,1) 亂數當作「等化後的符號」。
    """
    # I / Q 分別是獨立 Gaussian
    x_real = torch.randn(B, T, H, W, device=device)
    x_imag = torch.randn(B, T, H, W, device=device)
    # 疊成 C=2
    x = torch.stack([x_real, x_imag], dim=2)  # [B, T, 2, H, W]
    return x


def build_dummy_labels(
    B: int = 2, H: int = 32, W: int = 32, bits_per_symbol: int = 2, device="cpu"
) -> torch.Tensor:
    """
    隨機產生 bit label，形狀對齊 LLR 輸出：
      - shape: [B, H, W, bits_per_symbol]
      - 每個位置是 0 或 1
    """
    target = torch.randint(
        low=0, high=2, size=(B, H, W, bits_per_symbol), device=device, dtype=torch.float32
    )
    return target


def forward_detailed(model: SpikingRxModel, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    手動展開 SpikingRxModel 的 forward，方便印出每一層的 shape。

    流程：
      x0 = x                                   # [B,T,2,H,W]
      x1 = stem(x0)                            # [B,T,C1,H,W]
      x2 = stage1(x1)                          # [B,T,C2,H,W]
      ...
      x7 = stage6(x6)                          # [B,T,C7,H,W]
      rate = mean over time T                 # [B,C7,H,W]
      logits = readout(rate)                  # [B,2,H,W]
      llr = permute to [B,H,W,2]
    """
    B, T, C, H, W = x.shape

    # 1) StemConv
    x1 = model.stem(x)
    describe_tensor("After StemConv", x1)

    # 2) 6 個 SEW stages
    out = x1
    stage_acts = []
    for i, stage in enumerate(model.stages, start=1):
        out = stage(out)
        stage_acts.append(out)
        describe_tensor(f"After SEW Block {i}", out)

    # 3) 時間聚合（mean over T）→ spike rate 的 feature
    rate = out.clamp(0, 1).mean(dim=1)  # [B, C_last, H, W]
    describe_tensor("After time-mean rate", rate)

    # 4) Readout ANN
    logits = model.readout(rate) * model.llr_temperature  # [B, bits, H, W]
    describe_tensor("After Readout logits", logits)

    # 5) LLR 排成 [B, H, W, bits]
    llr = logits.permute(0, 2, 3, 1).contiguous()
    describe_tensor("Final LLR", llr)

    # 輔助資訊：每層 spike rate (照你的 SpikingRxModel forward 寫法)
    spike_rates = []
    with torch.no_grad():
        # 重新計算每個 stage 的 spike rate (僅示意，不再重跑整個網路 gradient)
        out_tmp = model.stem(x)
        for stage in model.stages:
            out_tmp = stage(out_tmp)           # [B,T,C,H,W]
            r_t = out_tmp.clamp(0, 1).mean(dim=(0, 2, 3, 4))  # [T]
            spike_rates.append(r_t.cpu())

    aux = {
        "spike_rate_per_stage": torch.stack(spike_rates, dim=0),  # [6, T]
        "final_rate_mean": rate.mean().detach().cpu(),
        "final_rate_std": rate.std().detach().cpu(),
    }
    return llr, aux


def train_spikingrx_demo(device="cpu"):
    """
    Demo 訓練流程：
      - 產生假 OFDM 輸入 (I/Q)
      - 建立 SpikingRxModel (in_ch=2，照論文)
      - 3 個 epoch：
          每次 forward → 算 loss → backward → update
      - 每個 epoch 印：
          loss
          final spike rate mean/std
          每 stage spike_rate_per_stage
      - 訓練前 / 訓練後各跑一次 forward_detailed 印整個架構的輸入輸出形狀
    """
    print(f"Using device: {device}")
    B, T, H, W = 2, 5, 32, 32
    bits_per_symbol = 2

    # -------------------------------------------------
    # 建 dummy 輸入 / 標籤
    # -------------------------------------------------
    x = build_dummy_ofdm_input(B=B, T=T, H=H, W=W, device=device)
    target = build_dummy_labels(B=B, H=H, W=W, bits_per_symbol=bits_per_symbol, device=device)

    print("\n=== Input Description ===")
    describe_tensor("Input x (I/Q OFDM)", x)
    print("維度對應：B=batch, T=time steps, C=2(I/Q), H=symbol index, W=subcarrier index")

    # -------------------------------------------------
    # 建立模型（照論文：in_ch=2）
    # -------------------------------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=bits_per_symbol,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
    ).to(device)

    # 簡單用 Adam + BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # -------------------------------------------------
    # 訓練前：看一次整個 forward 的形狀
    # -------------------------------------------------
    print("\n=== Forward Detail BEFORE Training ===")
    model.eval()
    with torch.no_grad():
        llr0, aux0 = forward_detailed(model, x)

    print("\nSpike rate per stage (BEFORE training):")
    for i, r_t in enumerate(aux0["spike_rate_per_stage"], start=1):
        print(f"  Stage {i}: {r_t.tolist()}")

    # -------------------------------------------------
    # 訓練：3 個 epoch（每個 epoch = 1 次 forward/backward）
    # -------------------------------------------------
    print("\n=== Training Start (3 epochs demo) ===")
    model.train()
    for epoch in range(1, 4):
        optimizer.zero_grad()
        llr, aux = model(x)   # 這裡用你原本 SpikingRxModel 的 forward

        # llr: [B,H,W,2]，target 同 shape
        loss = criterion(llr, target)
        loss.backward()
        optimizer.step()

        rate_mean = aux["final_rate_mean"].item()
        rate_std = aux["final_rate_std"].item()

        print(f"[Epoch {epoch}] loss={loss.item():.6f}  "
              f"rate_mean={rate_mean:.6f}  std={rate_std:.6f}")

        print("  Spike rate per stage (each is [T] over time):")
        sr = aux["spike_rate_per_stage"].detach().cpu()
        for i in range(sr.shape[0]):
            vals = ", ".join(f"{v:.4f}" for v in sr[i])
            print(f"    Stage {i+1}: [{vals}]")

    # -------------------------------------------------
    # 訓練後：再看一次整個 forward 的形狀與統計量
    # -------------------------------------------------
    print("\n=== Forward Detail AFTER Training (3 epochs) ===")
    model.eval()
    with torch.no_grad():
        llr1, aux1 = forward_detailed(model, x)

    print("\nSpike rate per stage (AFTER training):")
    for i, r_t in enumerate(aux1["spike_rate_per_stage"], start=1):
        print(f"  Stage {i}: {r_t.tolist()}")

    print(f"\nFinal LLR stats BEFORE training: mean={llr0.mean().item():.6f}, "
          f"std={llr0.std().item():.6f}")
    print(f"Final LLR stats AFTER  training: mean={llr1.mean().item():.6f}, "
          f"std={llr1.std().item():.6f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_spikingrx_demo(device=device)
