#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Path setup
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(SRC_DIR, "data")

for p in [SRC_DIR, DATA_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from models.spikingrx_model import SpikingRxModel
from data.oai_to_spikingrx_tensor import load_oai_fullgrid


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def read_int_kv(cfg_txt: Path, key: str):
    if not cfg_txt.exists():
        return None
    for ln in cfg_txt.read_text().splitlines():
        p = ln.strip().split()
        if len(p) == 2 and p[0] == key:
            return int(p[1])
    return None


def find_latest_fullgrid():
    base_dir = "/home/richard93513/SpikingRx-on-OAI/spx_records/raw"
    pattern = f"{base_dir}/f*_s*_fullgrid_idx*.bin"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No fullgrid dump found in: {base_dir}")
    return max(files, key=os.path.getmtime)


def infer_bundle_dir_from_dump(dump_path: Path):
    p = dump_path.resolve()
    if p.name == "fullgrid.bin" and p.parent.exists():
        return p.parent
    return None


def save_llr_heatmap(flat_llr: np.ndarray, out_png: Path, max_cols: int = 256):
    x = flat_llr.astype(np.float32).reshape(-1)
    n = x.size
    cols = min(max_cols, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    img = np.zeros((rows, cols), dtype=np.float32)
    img.flat[:n] = x

    plt.figure(figsize=(10, 4))
    plt.imshow(img, aspect="auto", cmap="bwr")
    plt.colorbar()
    plt.title("SpikingRx inferred LLR (flattened)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_spike_rate(sr_tensor, out_png: Path):
    sr_np = sr_tensor.detach().cpu().numpy()
    if sr_np.ndim == 1:
        mean_rate = sr_np
    else:
        mean_rate = sr_np.mean(axis=1)

    x = np.arange(1, len(mean_rate) + 1)

    plt.figure()
    plt.plot(x, mean_rate, marker="o")
    plt.xlabel("Stage")
    plt.ylabel("Mean spike rate")
    plt.title("Spike rate per stage")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def flatten_llr_to_f32(y_pred, flip_sign=False, target_length=None):
    llr = y_pred.detach().cpu().numpy().astype(np.float32).reshape(-1)

    if flip_sign:
        llr = -llr

    if target_length is not None:
        if llr.size < target_length:
            raise ValueError(
                f"[ERROR] inferred llr too short: {llr.size} < target_length={target_length}"
            )
        llr = llr[:target_length]

    return llr.astype(np.float32)


def extract_state_dict(ckpt_obj):
    """
    Support:
      1) pure state_dict
      2) {'state_dict': ...}
      3) other wrapped dicts
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    return ckpt_obj


def clean_state_dict_keys(state_dict):
    clean = {}
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace("_orig_mod.", "")
        nk = nk.replace("model.", "")
        clean[nk] = v
    return clean


def load_checkpoint_flexible(model, ckpt_path: Path):
    """
    1) always load checkpoint to CPU first (avoid OOM at torch.load)
    2) if keys mismatch, only load same-name same-shape params
    """
    print("Loading checkpoint (CPU first):", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = extract_state_dict(ckpt)
    state = clean_state_dict_keys(state)

    model_state = model.state_dict()
    matched = {}
    skipped = []

    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            matched[k] = v
        else:
            skipped.append(k)

    if not matched:
        raise RuntimeError(
            "No checkpoint parameters matched current model. "
            "Likely wrong checkpoint or wrong model definition."
        )

    missing_keys, unexpected_keys = model.load_state_dict(matched, strict=False)

    print(f"[CKPT] matched params   = {len(matched)}")
    print(f"[CKPT] skipped params   = {len(skipped)}")
    print(f"[CKPT] missing keys     = {len(missing_keys)}")
    print(f"[CKPT] unexpected keys  = {len(unexpected_keys)}")

    if skipped:
        print("[CKPT] first skipped keys:")
        for k in skipped[:10]:
            print("   ", k)

    if missing_keys:
        print("[CKPT] first missing keys:")
        for k in missing_keys[:10]:
            print("   ", k)

    return {
        "matched_params": len(matched),
        "skipped_params": len(skipped),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "first_skipped": skipped[:20],
    }


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run single-bundle SpikingRx inference and save infer_llr_f32.bin for check_oai_llr_decode.py"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/richard93513/SpikingRx-on-OAI/archive/checkpoints/best_spikingrx_model.BACKUP_20260209_161720.pth",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        help="Path to fullgrid.bin. If omitted, use latest raw fullgrid dump.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. If omitted and dump is bundle/fullgrid.bin, use that bundle dir.",
    )
    parser.add_argument(
        "--out-bin",
        type=str,
        default="infer_llr_f32.bin",
        help="Output float32 LLR binary filename",
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        default=False,
        help="Flip sign before saving infer_llr_f32.bin",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Force saved LLR length. If omitted, use G from ldpc_cfg.txt if available.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=3,
        help="Temporal steps. Default=3 to match prior train/debug usage.",
    )
    parser.add_argument(
        "--H-out",
        type=int,
        default=32,
        help="Output tensor height",
    )
    parser.add_argument(
        "--W-out",
        type=int,
        default=32,
        help="Output tensor width",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        default=False,
        help="Also save llr.npy",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=False,
        help="Also save llr heatmap / spike rate plots",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Decide devices
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        print("[Device] CUDA GPU =", torch.cuda.get_device_name(0))
        device_conv = torch.device("cuda")
    else:
        print("[Device] CPU only")
        device_conv = torch.device("cpu")

    device_fc = torch.device("cpu")

    # ---------------------------------------------------------
    # Dump path
    # ---------------------------------------------------------
    dump_path = Path(args.dump).resolve() if args.dump else Path(find_latest_fullgrid()).resolve()
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump not found: {dump_path}")
    print("Using dump:", dump_path)

    # ---------------------------------------------------------
    # Output dir
    # ---------------------------------------------------------
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    else:
        inferred_bundle_dir = infer_bundle_dir_from_dump(dump_path)
        if inferred_bundle_dir is not None:
            out_dir = inferred_bundle_dir
        else:
            out_dir = Path(CURRENT_DIR) / "out"

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir:", out_dir)

    # ---------------------------------------------------------
    # Read config / decide out_bits
    # ---------------------------------------------------------
    cfg_txt = out_dir / "ldpc_cfg.txt"
    g_from_cfg = read_int_kv(cfg_txt, "G")
    if g_from_cfg is None:
        raise RuntimeError(
            f"G not found. Expect ldpc_cfg.txt in output directory: {cfg_txt}"
        )

    out_bits = int(g_from_cfg)
    target_length = args.target_length if args.target_length is not None else out_bits

    print(f"[CFG] G={out_bits}")
    print(f"[CFG] target_length={target_length}")

    # ---------------------------------------------------------
    # Load fullgrid -> tensor
    # ---------------------------------------------------------
    x, meta = load_oai_fullgrid(
        str(dump_path),
        H_out=args.H_out,
        W_out=args.W_out,
        T=args.T,
        device=device_conv,
    )
    print("Dump meta:", meta)
    print("Input x shape:", tuple(x.shape))

    # ---------------------------------------------------------
    # Build model (match training/debug style)
    # ---------------------------------------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
        out_bits=out_bits,
        T=args.T,
        device_conv=device_conv,
        device_fc=device_fc,
    )

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_meta = load_checkpoint_flexible(model, ckpt_path)

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred, aux = model(x)

    print("Pred tensor shape:", tuple(y_pred.shape))
    print("Pred tensor device:", y_pred.device)

    # ---------------------------------------------------------
    # Save float32 LLR for decode pipeline
    # ---------------------------------------------------------
    llr_f32 = flatten_llr_to_f32(
        y_pred=y_pred,
        flip_sign=args.flip_sign,
        target_length=target_length,
    )

    out_bin = out_dir / args.out_bin
    llr_f32.tofile(out_bin)

    print(f"[OK] Saved infer LLR f32 -> {out_bin}")
    print(f"     length = {llr_f32.size}")
    print(f"     min    = {float(llr_f32.min()):.6f}")
    print(f"     max    = {float(llr_f32.max()):.6f}")
    print(f"     mean   = {float(llr_f32.mean()):.6f}")
    print(f"     std    = {float(llr_f32.std()):.6f}")

    # ---------------------------------------------------------
    # Optional saves
    # ---------------------------------------------------------
    if args.save_npy:
        llr_npy = out_dir / "llr.npy"
        np.save(llr_npy, y_pred.detach().cpu().numpy())
        print("Saved:", llr_npy)

    if args.save_plots:
        try:
            llr_png = out_dir / "llr_heatmap.png"
            save_llr_heatmap(llr_f32, llr_png)
            print("Saved:", llr_png)
        except Exception as e:
            print("[WARN] Failed to save llr heatmap:", e)

        if isinstance(aux, dict) and "spike_rate_per_stage" in aux:
            try:
                sr = aux["spike_rate_per_stage"]
                sr_png = out_dir / "spike_rate.png"
                save_spike_rate(sr, sr_png)
                print("Saved:", sr_png)
            except Exception as e:
                print("[WARN] Failed to save spike rate plot:", e)

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------
    meta_out = {
        "dump_path": str(dump_path),
        "out_dir": str(out_dir),
        "out_bin": str(out_bin),
        "checkpoint": str(ckpt_path),
        "device_conv": str(device_conv),
        "device_fc": str(device_fc),
        "flip_sign": args.flip_sign,
        "T": args.T,
        "out_bits": out_bits,
        "target_length": target_length,
        "pred_shape": list(y_pred.shape),
        "llr_saved_len": int(llr_f32.size),
        "llr_stats": {
            "min": float(llr_f32.min()),
            "max": float(llr_f32.max()),
            "mean": float(llr_f32.mean()),
            "std": float(llr_f32.std()),
        },
        "checkpoint_load": ckpt_meta,
        "dump_meta": meta,
    }

    meta_path = out_dir / "spikingrx_infer_meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
    print("Saved:", meta_path)

    print()
    print("Next step:")
    print(
        f'python3 ~/SpikingRx-on-OAI/src/inference/check_oai_llr_decode.py '
        f'{out_bin} '
        f'{out_dir / "ldpc_cfg.txt"} '
        f'{out_dir / "pdsch_cfg.txt"} '
        f'{out_dir / "txbits.bin"} '
        f'--rmunmatch ~/openairinterface5g/cmake_targets/ran_build/build/rmunmatch_spx '
        f'--ldpctest ~/openairinterface5g/cmake_targets/ran_build/build/ldpctest_spx '
        f'--llr-scale 1 '
        f'--out decoded_bits.bin'
    )


if __name__ == "__main__":
    main()
