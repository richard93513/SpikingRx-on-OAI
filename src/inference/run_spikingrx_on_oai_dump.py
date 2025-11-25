# src/tests/run_spikingrx_on_oai_dump.py
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
def save_llr_for_oai_decoder(llr_tensor, out_path,
                             llr_clip=8.0,
                             flip_sign=True,
                             target_length=None,
                             verbose=True):
    """
    將 SpikingRx LLR (float32, shape [1, H, W, bits]) 轉成 OAI 能吃的 int8 (length = G)。
    - llr_clip: 正規化係數，LLR / llr_clip → [-1, 1]
    - target_length: 必須等於 OAI PDSCH 的 G（coded bits 數量）
    """

    llr = llr_tensor.detach().cpu().numpy()  # shape [1, H, W, bits]
    llr = llr.astype(np.float32).reshape(-1)  # flatten

    if flip_sign:
        llr = -llr

    # normalize to [-1,1]
    llr_norm = llr / llr_clip
    llr_norm = np.clip(llr_norm, -1.0, 1.0)

    # scale → [-127,127]
    llr_int8 = np.round(llr_norm * 127).astype(np.int8)

    # length check
    if target_length is not None:
        if llr_int8.shape[0] < target_length:
            raise ValueError(f"[ERROR] LLR bits too short: {llr_int8.shape[0]} < G={target_length}")
        if llr_int8.shape[0] > target_length:
            # 截斷前 G 個 bit
            llr_int8 = llr_int8[:target_length]

    # save binary
    llr_int8.tofile(out_path)

    if verbose:
        print(f"[OK] Saved OAI LLR → {out_path}")
        print(f"     length = {llr_int8.shape[0]}")
        print(f"     min = {llr_int8.min()}, max = {llr_int8.max()}")

    return llr_int8

    """
    將 SpikingRx LLR（float32）轉成 OAI LDPC decoder 能吃的 int8。
    """
    llr = llr_tensor.detach().cpu().numpy().astype(np.float32)

    # 1) 通常 SNN 的 LLR 定義是 log(P1/P0)，OAI 要 log(P0/P1)
    if flip_sign:
        llr = -llr

    # 2) normalize → [-1,1]
    llr = llr / max_abs_llr
    llr = np.clip(llr, -1.0, 1.0)

    # 3) scale → [-127,127] int8
    llr_int8 = np.round(llr * 127).astype(np.int8)

    # 4) flatten
    flat = llr_int8.reshape(-1)

    # 5) save
    flat.tofile(out_path)
    print(f"[OK] LLR saved for OAI decoder → {out_path}, {flat.shape[0]} bits")


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
    parser.add_argument("--ckpt", type=str, default="/home/richard93513/SpikingRx-on-OAI/src/train/spikingrx_checkpoint.pth",
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

    # --- NEW: Save LLR for OAI LDPC decoder ---

    # 從 metadata 取得 OAI 的 coded bits 數量 G
    # 若沒有 G，就用 N_re * bits_per_symbol (QPSK → ×2)
    
    # --- Decide final LLR length ---
    # NOTE: 目前暫時不依賴 OAI 的 G，直接 flatten SpikingRx LLR
    llr_flat = llr.detach().cpu().numpy().reshape(-1)
    G = llr_flat.shape[0]   # ex: 2048 bits


    llr_bin = os.path.join(out_dir, "llr_int8_for_oai.bin")

    save_llr_for_oai_decoder(
        llr,
        llr_bin,
        llr_clip=8.0,
        flip_sign=True,
        target_length=G,
    )

    print("Saved:", llr_bin)



if __name__ == "__main__":
    main()

