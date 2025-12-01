# src/inference/batch_inference_on_bundle.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

# -----------------------------------------------
#  Import SpikingRx + loader
# -----------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
DATA_DIR = os.path.join(SRC_DIR, "data")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from models.spikingrx_model import SpikingRxModel
from data.oai_to_spikingrx_tensor import load_oai_fullgrid


# -----------------------------------------------
#   LLR → OAI format (int8)
# -----------------------------------------------
def save_llr_for_oai_decoder(
    llr_tensor,
    out_path,
    llr_clip=8.0,
    flip_sign=True,
    target_length=None,
):
    llr = llr_tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)

    if flip_sign:
        llr = -llr

    llr_norm = np.clip(llr / llr_clip, -1.0, 1.0)
    llr_int8 = np.round(llr_norm * 127).astype(np.int8)

    if target_length is not None:
        llr_int8 = llr_int8[:target_length]

    llr_int8.tofile(out_path)
    return llr_int8


# -----------------------------------------------
#   Visualization
# -----------------------------------------------
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
    sr_np = sr.detach().cpu().numpy()  # [stage, T]
    mean_rate = sr_np.mean(axis=1)
    x = np.arange(1, sr_np.shape[0] + 1)

    fig, ax = plt.subplots()
    ax.plot(x, mean_rate, marker="o")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mean spike rate")
    ax.set_title("Spike rate per stage")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


# -----------------------------------------------
#   MAIN: batch inference
# -----------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle_root",
        type=str,
        default="/home/richard93513/SpikingRx-on-OAI/spx_records/bundle",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/richard93513/SpikingRx-on-OAI/src/train/spikingrx_checkpoint.pth",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # -------------------------------------------------------
    #  Load SpikingRx model
    # -------------------------------------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
    ).to(device)

    if os.path.exists(args.ckpt):
        print(f"[Load] Checkpoint: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
    else:
        print("⚠ No checkpoint found → using random weights")

    model.eval()

    # -------------------------------------------------------
    #  Scan bundles
    # -------------------------------------------------------
    bundle_dirs = sorted(glob.glob(f"{args.bundle_root}/f*_s*"))

    print(f"[INFO] Found {len(bundle_dirs)} bundles")

    for bdir in bundle_dirs:
        print("------------------------------------------------------")
        print(f"[Bundle] {bdir}")

        fgrid = os.path.join(bdir, "fullgrid.bin")
        ldpc_cfg = os.path.join(bdir, "ldpc_cfg.json")

        if not os.path.exists(fgrid):
            print("  → No fullgrid.bin, skip")
            continue

        with open(os.path.join(bdir, "meta.json"), "r") as f:
            meta = json.load(f)

        # -------------------------------------------------------
        # Load LDPC config (for G length)
        # -------------------------------------------------------
        if os.path.exists(ldpc_cfg):
            with open(ldpc_cfg, "r") as f:
                cfg = json.load(f)
            G = int(cfg["G"])
        else:
            print("  ⚠ No ldpc_cfg.json, cannot determine G → skip")
            continue

        # -------------------------------------------------------
        # Load fullgrid → tensor
        # -------------------------------------------------------
        x, fg_meta = load_oai_fullgrid(
            fgrid,
            H_out=32,
            W_out=32,
            T=5,
            device=device,
        )

        # -------------------------------------------------------
        # Run inference
        # -------------------------------------------------------
        with torch.no_grad():
            llr, aux = model(x)

        # -------------------------------------------------------
        # Save outputs into same bundle dir
        # -------------------------------------------------------
        llr_float_path = os.path.join(bdir, "infer_llr_float.npy")
        np.save(llr_float_path, llr.detach().cpu().numpy())

        llr_int8_path = os.path.join(bdir, "infer_llr_int8.bin")
        llr_int8 = save_llr_for_oai_decoder(
            llr,
            llr_int8_path,
            llr_clip=8.0,
            flip_sign=True,
            target_length=G,
        )

        # heatmap
        heat_png = os.path.join(bdir, "llr_heatmap.png")
        save_llr_heatmap(llr, heat_png)

        # spike rate
        if isinstance(aux, dict) and "spike_rate_per_stage" in aux:
            sr_png = os.path.join(bdir, "spike_rate.png")
            save_spike_rate(aux["spike_rate_per_stage"], sr_png)

        # -------------------------------------------------------
        # Save inference meta
        # -------------------------------------------------------
        infer_meta = {
            "frame": meta["frame"],
            "slot": meta["slot"],
            "fg_idx": meta["fg_idx"],
            "model_ckpt": args.ckpt,
            "input_fullgrid": fgrid,
            "output_llr_float": llr_float_path,
            "output_llr_int8": llr_int8_path,
            "G": G,
        }

        with open(os.path.join(bdir, "infer_meta.json"), "w") as f:
            json.dump(infer_meta, f, indent=2)

        print(f"  → Inference done, G={G}")
        print(f"  → Saved infer_meta.json")


# -----------------------------------------------
if __name__ == "__main__":
    main()

