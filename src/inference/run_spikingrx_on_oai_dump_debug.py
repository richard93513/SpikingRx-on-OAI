# src/tests/run_spikingrx_on_oai_dump_debug.py
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


def dump_tensor_stats(name, x):
    x = x.detach().cpu()
    print(f"{name}: shape={tuple(x.shape)}, mean={x.mean():.6f}, std={x.std():.6f}")


def find_latest(pattern="/tmp/spx_fullgrid_f*_s*.bin"):
    files = glob.glob(pattern)
    assert files, f"No dump found: {pattern}"
    return max(files, key=os.path.getmtime)


def save_llr_heatmap(llr, out_png):
    llr_img = llr.detach().cpu().numpy()[0, ..., 0]
    plt.imshow(llr_img, cmap="bwr")
    plt.colorbar()
    plt.title("LLR Heatmap (bit0)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_spike_rate(sr, out_png):
    sr = sr.detach().cpu().numpy()
    plt.plot(np.arange(1, len(sr) + 1), sr, marker="o")
    plt.title("Spike Rate per Stage (time-avg)")
    plt.xlabel("Stage")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="../train/spikingrx_checkpoint.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dump_path = find_latest()
    print("Using dump:", dump_path)

    x, meta = load_oai_fullgrid(
        dump_path,
        H_out=32,
        W_out=32,
        T=5,
        n_rb=106,
        device=device,
    )
    dump_tensor_stats("Input x", x)

    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0
    ).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        print("Loading checkpoint:", args.ckpt)
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print("⚠ No checkpoint loaded — using random weights")

    model.eval()
    with torch.no_grad():
        out = model.stem(x)
        dump_tensor_stats("Stem output", out)

        spike_rates_stages = []

        for i, stage in enumerate(model.stages):
            out = stage(out)
            dump_tensor_stats(f"Stage {i+1} output", out)
            r = out.clamp(0, 1).mean(dim=(0, 2, 3, 4))
            spike_rates_stages.append(r.detach().cpu())

        rate = out.clamp(0, 1).mean(dim=1)
        dump_tensor_stats("Final rate (time-avg)", rate)

        logits = model.readout(rate) * model.llr_temperature
        llr = logits.permute(0, 2, 3, 1).contiguous()
        dump_tensor_stats("LLR output", llr)

    out_dir = os.path.join(CURRENT_DIR, "out_debug")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "llr.npy"), llr.cpu().numpy())
    print("Saved: llr.npy")

    save_llr_heatmap(llr, os.path.join(out_dir, "llr_heatmap.png"))
    print("Saved: llr_heatmap.png")

    sr_stack = torch.stack(spike_rates_stages)  # [S, T]
    mean_rate = sr_stack.mean(dim=1)
    save_spike_rate(mean_rate, os.path.join(out_dir, "spike_rate.png"))
    print("Saved: spike_rate.png")


if __name__ == "__main__":
    main()

