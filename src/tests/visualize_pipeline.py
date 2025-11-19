# -*- coding: utf-8 -*-
"""
視覺化 OAI → Full-grid → SpikingRx 的整個 pipeline
用來做專題簡報、資料比較、展示系統每層的行為。
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
DATA_DIR = os.path.join(SRC_DIR, "data")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from models.spikingrx_model import SpikingRxModel
from data.oai_to_spikingrx_tensor import load_oai_fullgrid


def find_latest_fullgrid(pattern="/tmp/spx_fullgrid_f*_s*.bin"):
    files = glob.glob(pattern)
    assert files, f"No dump files found under {pattern}"
    return max(files, key=os.path.getmtime)


def show_heatmap(mat, title, savepath):
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def main():
    out_dir = os.path.join(CURRENT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------
    # 1) 找最新 FULL-GRID 檔案
    # ------------------------------
    dump_path = find_latest_fullgrid()
    print(f"Using dump file: {dump_path}")

    # ------------------------------
    # 2) 讀 FULL-GRID → 映射成 32×32
    # ------------------------------
    x, meta = load_oai_fullgrid(
        dump_path,
        H_out=32,
        W_out=32,
        T=5,
        n_rb=106,
        device=device,
    )

    print("\n======= Dump Meta =======")
    for k, v in meta.items():
        print(f"{k}: {v}")

    # x: (1, T, 2, 32, 32)
    x_np = x.cpu().numpy()

    # ------------------------------
    # 3) 視覺化輸入：I/Q heatmap
    # ------------------------------
    CH_img = x_np[0, 0]     # (C=2, H=32, W=32)

    show_heatmap(CH_img[0], "Input: I-channel (32×32)", 
                 os.path.join(out_dir, "input_I.png"))
    show_heatmap(CH_img[1], "Input: Q-channel (32×32)", 
                 os.path.join(out_dir, "input_Q.png"))

    print("\nSaved input heatmaps.")

    # ------------------------------
    # 4) 建立模型（未訓練，也可以換成 checkpoint）
    # ------------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
    ).to(device)

    ckpt = os.path.join(CURRENT_DIR, "spikingrx_checkpoint.pth")
    if os.path.exists(ckpt):
        print("Loading checkpoint...")
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print("⚠ No checkpoint found, using random model.")

    model.eval()
    with torch.no_grad():
        llr, aux = model(x)

    # ------------------------------
    # 5) Spike rate per stage (bar chart)
    # ------------------------------
    sr = aux["spike_rate_per_stage"].cpu().numpy()   # shape: (num_stage, T)

    plt.figure(figsize=(7, 4))
    for i, stage_sr in enumerate(sr):
        plt.plot(stage_sr, marker="o", label=f"Stage {i+1}")
    plt.title("Spike Rate per Stage (T=5)")
    plt.xlabel("Time step")
    plt.ylabel("Spike rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spike_rate_stages.png"))
    plt.close()

    print("Saved spike rate plot.")

    # ------------------------------
    # 6) LLR heatmap（bit0, bit1）
    # ------------------------------
    llr_np = llr.cpu().numpy()[0]   # (H=32, W=32, 2)
    llr0 = llr_np[:, :, 0]
    llr1 = llr_np[:, :, 1]

    show_heatmap(llr0, "LLR Bit 0", os.path.join(out_dir, "llr_bit0.png"))
    show_heatmap(llr1, "LLR Bit 1", os.path.join(out_dir, "llr_bit1.png"))

    print("Saved LLR heatmaps.")

    # ------------------------------
    # 7) 數值統計比較
    # ------------------------------
    print("\n======= Statistics =======")
    print(f"Input mean={x.mean():.6f}, std={x.std():.6f}")
    print(f"LLR   mean={llr.mean():.6f}, std={llr.std():.6f}")

    print("\nAll results saved to:", out_dir)


if __name__ == "__main__":
    main()

