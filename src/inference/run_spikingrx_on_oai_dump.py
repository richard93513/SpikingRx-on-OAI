# src/tests/run_spikingrx_on_oai_dump.py
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def save_llr_heatmap(llr, out_png):
    llr_np = llr.detach().cpu().numpy()[0]  # (32,32,2)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for b in range(2):
        im = axs[b].imshow(llr_np[..., b], cmap="bwr")
        axs[b].set_title(f"LLR bit{b}")
        plt.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def save_spike_rate(sr, out_png):
    # sr: [num_stages, T]
    sr_np = sr.detach().cpu().numpy()
    stages, T = sr_np.shape
    x = np.arange(1, stages + 1)
    mean_rate = sr_np.mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(x, mean_rate, marker="o")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mean spike rate")
    ax.set_title("Spike rate per stage (time-avg)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="../train/spikingrx_checkpoint.pth",
                        help="path to trained checkpoint (state_dict)")
    parser.add_argument("--dump", type=str, default=None,
                        help="OAI full-grid dump path, default: latest /tmp/spx_fullgrid_*.bin")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dump_path = args.dump if args.dump is not None else find_latest()
    print("Using dump:", dump_path)

    x, meta = load_oai_fullgrid(
        dump_path,
        H_out=32,
        W_out=32,
        T=5,
        n_rb=106,
        device=device,
    )
    print("Dump meta:", meta)
    print("Input x shape:", tuple(x.shape))

    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
    ).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        print("Loading checkpoint:", args.ckpt)
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print("⚠ No valid checkpoint, using random weights")

    model.eval()
    with torch.no_grad():
        llr, aux = model(x)

    out_dir = os.path.join(CURRENT_DIR, "out")
    os.makedirs(out_dir, exist_ok=True)

    llr_npy = os.path.join(out_dir, "llr.npy")
    np.save(llr_npy, llr.detach().cpu().numpy())
    print("Saved:", llr_npy)

    llr_png = os.path.join(out_dir, "llr_heatmap.png")
    save_llr_heatmap(llr, llr_png)
    print("Saved:", llr_png)

    if isinstance(aux, dict) and "spike_rate_per_stage" in aux:
        sr = aux["spike_rate_per_stage"]  # [S, T]
        sr_png = os.path.join(out_dir, "spike_rate.png")
        save_spike_rate(sr, sr_png)
        print("Saved:", sr_png)


if __name__ == "__main__":
    main()
