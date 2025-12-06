# src/visualize/visualize_spiking_activity.py
# -*- coding: utf-8 -*-

"""
SpikingRx Visualization Tool (updated for OAI integration)
----------------------------------------------------------
功能：
1. Spike rate before vs after training
2. 每個 stage 的 spike GIF
3. LLR 分布圖（新版，不使用 2D heatmap）

Checkpoint: 自動載入 best_spikingrx_model.pth
Input     : bundle or raw fullgrid dump
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
DATA_DIR = os.path.join(SRC_DIR, "data")

for p in [SRC_DIR, DATA_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from models.spikingrx_model import SpikingRxModel
from data.oai_to_spikingrx_tensor import load_oai_fullgrid


# ------------------------------------------------
#  讀取最新 fullgrid dump（支援 bundle）
# ------------------------------------------------
def find_latest_fullgrid():
    patterns = [
        "/tmp/spx_fullgrid_f*_s*.bin",
        str(os.path.expanduser("~/SpikingRx-on-OAI/spx_records/bundle/f*/fullgrid.bin"))
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    assert files, "❌ 找不到 fullgrid dump"
    return max(files, key=os.path.getmtime)


# ------------------------------------------------
#   收集每層 SEW block 的輸出（for GIF）
# ------------------------------------------------
def forward_collect_stages(model, x, device):
    model.eval()
    with torch.no_grad():
        out = model.stem(x)
        stage_outputs = []
        for stage in model.stages:
            out = stage(out)
            stage_outputs.append(out.detach().cpu())  # [B,T,C,H,W]
    return stage_outputs


# ------------------------------------------------
#   spike rate per stage
# ------------------------------------------------
def compute_spike_rate_per_stage(model, x):
    model.eval()
    with torch.no_grad():
        _, aux = model(x)
    return aux["spike_rate_per_stage"].detach().cpu()  # [S, T]


# ------------------------------------------------
#   GIF animation for spiking activity
# ------------------------------------------------
def make_spike_gif(stage_tensor, stage_idx, out_dir):
    B, T, C, H, W = stage_tensor.shape
    frames = [stage_tensor[0, t].mean(dim=0).numpy() for t in range(T)]

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="Greys", vmin=0.0, vmax=1.0)

    def update(i):
        im.set_data(frames[i])
        ax.set_title(f"Stage {stage_idx+1}, t={i}")
        return [im]

    ani = FuncAnimation(fig, update, frames=T, blit=True)
    os.makedirs(out_dir, exist_ok=True)
    out_gif = os.path.join(out_dir, f"spike_stage{stage_idx+1}.gif")
    ani.save(out_gif, writer=PillowWriter(fps=2))
    plt.close(fig)
    print("Saved:", out_gif)


# ------------------------------------------------
#    spike rates 比較圖
# ------------------------------------------------
def plot_spike_rate_compare(sr_before, sr_after, out_png):
    before = sr_before.mean(dim=1).numpy()
    after = sr_after.mean(dim=1).numpy()
    x = np.arange(1, len(before) + 1)

    fig, ax = plt.subplots()
    ax.plot(x, before, marker="o", label="random init")
    ax.plot(x, after, marker="s", label="trained")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mean spike rate")
    ax.set_title("Spike rate per stage (before vs after training)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print("Saved:", out_png)


# ------------------------------------------------
#   LLR distribution plot
# ------------------------------------------------
def plot_llr_distribution(llr, out_png):
    llr_np = llr.detach().cpu().numpy().reshape(-1)
    plt.figure(figsize=(10, 4))
    plt.hist(llr_np, bins=100, color="b", alpha=0.7)
    plt.title("LLR Distribution (14400 dims)")
    plt.xlabel("LLR value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("Saved:", out_png)


# ============================================================
#   MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/richard93513/SpikingRx-on-OAI/src/train/best_spikingrx_model.pth",
        help="trained checkpoint path"
    )
    parser.add_argument("--dump", type=str, default=None, help="Fullgrid dump path")
    args = parser.parse_args()

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # -------------------------
    # Load fullgrid
    # -------------------------
    dump = args.dump if args.dump else find_latest_fullgrid()
    print("Using dump:", dump)

    x, meta = load_oai_fullgrid(
        dump,
        H_out=32,
        W_out=32,
        T=5,
        n_rb=106,
        device=device,
    )

    # -------------------------
    # Random model (baseline)
    # -------------------------
    model_before = SpikingRxModel(
        in_ch=2, base_ch=16, bits_per_symbol=2,
        beta=0.9, theta=0.5, llr_temperature=1.0
    ).to(device)

    # -------------------------
    # Trained model
    # -------------------------
    model_after = SpikingRxModel(
        in_ch=2, base_ch=16, bits_per_symbol=2,
        beta=0.9, theta=0.5, llr_temperature=1.0
    ).to(device)

    if os.path.exists(args.ckpt):
        print("Loading trained checkpoint:", args.ckpt)
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        model_after.load_state_dict(state)
    else:
        print("⚠ No trained checkpoint, using random model instead.")

    # -------------------------
    # Output folder
    # -------------------------
    out_dir = os.path.join(CURRENT_DIR, "out_vis")
    os.makedirs(out_dir, exist_ok=True)

    # spike rate before/after
    sr_before = compute_spike_rate_per_stage(model_before, x)
    sr_after = compute_spike_rate_per_stage(model_after, x)

    plot_spike_rate_compare(sr_before, sr_after,
                            os.path.join(out_dir, "spike_rate_before_after.png"))

    # GIF for each stage
    stage_outputs = forward_collect_stages(model_after, x, device)
    for i, stage_tensor in enumerate(stage_outputs):
        make_spike_gif(stage_tensor, i, out_dir)

    # LLR distribution
    model_after.eval()
    with torch.no_grad():
        llr_pred, _ = model_after(x)
    plot_llr_distribution(
        llr_pred, os.path.join(out_dir, "llr_distribution_trained.png")
    )


if __name__ == "__main__":
    main()


