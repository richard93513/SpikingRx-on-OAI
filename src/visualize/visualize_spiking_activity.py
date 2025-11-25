# src/visualize/visualize_spiking_activity.py
# -*- coding: utf-8 -*-

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

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from models.spikingrx_model import SpikingRxModel
from data.oai_to_spikingrx_tensor import load_oai_fullgrid


def find_latest(pattern="/tmp/spx_fullgrid_f*_s*.bin"):
    files = glob.glob(pattern)
    assert files, f"No dump found: {pattern}"
    return max(files, key=os.path.getmtime)


def forward_collect_stages(model, x, device):
    model.eval()
    with torch.no_grad():
        out = model.stem(x)
        stage_outputs = []
        for stage in model.stages:
            out = stage(out)
            stage_outputs.append(out.detach().cpu())  # [B,T,C,H,W]
    return stage_outputs  # list of len=6


def compute_spike_rate_per_stage(model, x):
    model.eval()
    with torch.no_grad():
        _, aux = model(x)
    sr = aux["spike_rate_per_stage"].detach().cpu()  # [S, T]
    return sr


def make_spike_gif(stage_tensor, stage_idx, out_dir):
    # stage_tensor: [B, T, C, H, W]
    B, T, C, H, W = stage_tensor.shape
    frames = []
    for t in range(T):
        frame = stage_tensor[0, t].mean(dim=0)  # [H,W]
        frames.append(frame.numpy())

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_title(f"Stage {stage_idx+1}, t=0")

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


def plot_spike_rate_compare(sr_before, sr_after, out_png):
    # sr_*: [S, T]
    before = sr_before.mean(dim=1).numpy()
    after = sr_after.mean(dim=1).numpy()
    x = np.arange(1, len(before) + 1)

    fig, ax = plt.subplots()
    ax.plot(x, before, marker="o", label="random init")
    ax.plot(x, after, marker="s", label="trained")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mean spike rate")
    ax.set_title("Spike rate per stage (before vs after training)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print("Saved:", out_png)


def save_llr_heatmap(llr, out_png):
    llr_np = llr.detach().cpu().numpy()[0]
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for b in range(2):
        im = axs[b].imshow(llr_np[..., b], cmap="bwr")
        axs[b].set_title(f"LLR bit{b}")
        plt.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print("Saved:", out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/richard93513/SpikingRx-on-OAI/src/train/spikingrx_checkpoint.pth",
                        help="trained checkpoint path")
    parser.add_argument("--dump", type=str, default=None,
                        help="OAI dump path, default latest")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dump_path = args.dump if args.dump else find_latest()
    print("Using dump:", dump_path)

    x, meta = load_oai_fullgrid(
        dump_path,
        H_out=32,
        W_out=32,
        T=5,
        n_rb=106,
        device=device,
    )

    # random baseline
    model_before = SpikingRxModel(
        in_ch=2, base_ch=16, bits_per_symbol=2,
        beta=0.9, theta=0.5, llr_temperature=1.0
    ).to(device)

    # trained model
    model_after = SpikingRxModel(
        in_ch=2, base_ch=16, bits_per_symbol=2,
        beta=0.9, theta=0.5, llr_temperature=1.0
    ).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        print("Loading trained checkpoint:", args.ckpt)
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        model_after.load_state_dict(state)
    else:
        print("⚠ No trained checkpoint, both models are random")

    out_dir = os.path.join(CURRENT_DIR, "out_vis")
    os.makedirs(out_dir, exist_ok=True)

    # spike rate before / after
    sr_before = compute_spike_rate_per_stage(model_before, x)
    sr_after = compute_spike_rate_per_stage(model_after, x)
    plot_spike_rate_compare(sr_before, sr_after,
                            os.path.join(out_dir, "spike_rate_before_after.png"))

    # spike animations for trained model
    stage_outputs = forward_collect_stages(model_after, x, device)
    for i, stage_tensor in enumerate(stage_outputs):
        make_spike_gif(stage_tensor, i, out_dir)

    # LLR heatmap (trained)
    model_after.eval()
    with torch.no_grad():
        llr, _ = model_after(x)
    save_llr_heatmap(llr, os.path.join(out_dir, "llr_heatmap_trained.png"))


if __name__ == "__main__":
    main()

