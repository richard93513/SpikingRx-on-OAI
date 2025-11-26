# src/inference/run_spikingrx_on_oai_dump.py
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# -------------------------------------------------------------
#  LLR → OAI LDPC decoder int8 format
# -------------------------------------------------------------
def save_llr_for_oai_decoder(
    llr_tensor,
    out_path,
    llr_clip=8.0,
    flip_sign=True,
    target_length=None,
    verbose=True,
):
    # llr shape: [1, H, W, bits] or [1, *, *]
    llr = llr_tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)

    # OAI expects log(P0/P1)
    if flip_sign:
        llr = -llr

    # normalize → [-1, 1]
    llr_norm = np.clip(llr / llr_clip, -1.0, 1.0)

    # scale → int8 [-127,127]
    llr_int8 = np.round(llr_norm * 127).astype(np.int8)

    # adjust length
    if target_length is not None:
        if llr_int8.shape[0] < target_length:
            raise ValueError(
                f"[ERROR] LLR too short: {llr_int8.shape[0]} < G={target_length}"
            )
        llr_int8 = llr_int8[:target_length]

    llr_int8.tofile(out_path)

    if verbose:
        print(f"[OK] Saved OAI LLR → {out_path}")
        print(f"     length = {llr_int8.shape[0]}")
        print(f"     min={llr_int8.min()}, max={llr_int8.max()}")

    return llr_int8


# -------------------------------------------------------------
#  Path setup
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
DATA_DIR = os.path.join(SRC_DIR, "data")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

# -------------------------------------------------------------
#  Import SpikingRx model & loader
# -------------------------------------------------------------
from models.spikingrx_model import SpikingRxModel
from data.oai_to_spikingrx_tensor import load_oai_fullgrid


# -------------------------------------------------------------
#  找最新 fullgrid dump
# -------------------------------------------------------------
def find_latest():
    base_dir = "/home/richard93513/SpikingRx-on-OAI/spx_records/raw"
    pattern = f"{base_dir}/f*_s*_fullgrid_idx*.bin"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No fullgrid dump in: {base_dir}")
    return max(files, key=os.path.getmtime)


# -------------------------------------------------------------
#  儲存 heatmap
# -------------------------------------------------------------
def save_llr_heatmap(llr, out_png):
    # llr: [1, 32,32,2] or [1,T,...]
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
    # sr: [stages, T]
    sr_np = sr.detach().cpu().numpy()
    mean_rate = sr_np.mean(axis=1)
    x = np.arange(1, sr_np.shape[0] + 1)

    fig, ax = plt.subplots()
    ax.plot(x, mean_rate, marker="o")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mean spike rate")
    ax.set_title("Spike rate per stage (time-avg)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


# -------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/richard93513/SpikingRx-on-OAI/src/train/spikingrx_checkpoint.pth",
    )
    parser.add_argument("--dump", type=str, default=None)
    args = parser.parse_args()

    # 0) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Fullgrid dump path
    dump_path = args.dump if args.dump else find_latest()
    print("Using dump:", dump_path)

    # 2) Load tensor
    x, meta = load_oai_fullgrid(
        dump_path,
        H_out=32,
        W_out=32,
        T=5,
        device=device,
    )
    print("Dump meta:", meta)
    print("Input x shape:", tuple(x.shape))

    # 3) Model
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
    ).to(device)

    # 4) Load checkpoint
    if args.ckpt and os.path.exists(args.ckpt):
        print("Loading checkpoint:", args.ckpt)
        state = torch.load(args.ckpt, map_location=device)

        # 修正 Lightning / compile() 的 _orig_mod.prefix
        clean_state = {}
        for k, v in state.items():
            clean_state[k.replace("_orig_mod.", "")] = v

        model.load_state_dict(clean_state, strict=False)
    else:
        print("⚠ No checkpoint found, using random weights")

    # 5) Inference
    model.eval()
    with torch.no_grad():
        llr, aux = model(x)

    # 6) Output dir
    out_dir = os.path.join(CURRENT_DIR, "out")
    os.makedirs(out_dir, exist_ok=True)

    # Save LLR npy
    llr_npy = os.path.join(out_dir, "llr.npy")
    np.save(llr_npy, llr.detach().cpu().numpy())
    print("Saved:", llr_npy)

    # Save heatmap
    llr_png = os.path.join(out_dir, "llr_heatmap.png")
    save_llr_heatmap(llr, llr_png)
    print("Saved:", llr_png)

    # Save spike rate
    if isinstance(aux, dict) and "spike_rate_per_stage" in aux:
        sr = aux["spike_rate_per_stage"]
        sr_png = os.path.join(out_dir, "spike_rate.png")
        save_spike_rate(sr, sr_png)
        print("Saved:", sr_png)

    # 7) LLR → OAI LDPC decoder format
    flat = llr.detach().cpu().numpy().reshape(-1)
    G = flat.shape[0]  # 暫時直接當最大長度

    out_llr_bin = os.path.join(out_dir, "llr_int8_for_oai.bin")

    save_llr_for_oai_decoder(
        llr,
        out_llr_bin,
        llr_clip=8.0,
        flip_sign=True,
        target_length=G,
    )

    print("Saved:", out_llr_bin)


# -------------------------------------------------------------
if __name__ == "__main__":
    main()


