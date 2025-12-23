#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
單一檔案完成：
1. fullgrid → SpikingRx inference (T=3, base_ch=16)
2. 輸出 infer_llr_int8.bin
3. 呼叫 OAI ldpctest_spx 解碼 → decoded_bits.bin
4. 拿 txbits.bin 比對 → 計算 BER
"""

import os
import sys
import json
import glob
import numpy as np
import torch
import subprocess
from pathlib import Path
import argparse

# ----------------------------------------------------
# Import 模型與 tensor loader
# ----------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(CURRENT_DIR, "..")
DATA_DIR    = os.path.join(SRC_DIR, "data")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from models.spikingrx_model import SpikingRxModel
from data.oai_to_spikingrx_tensor import load_oai_fullgrid


# ----------------------------------------------------
# LLR → int8 給 OAI decoder
# ----------------------------------------------------
def save_llr_for_oai_decoder(llr_tensor, out_path, llr_clip, flip_sign, G):
    llr = llr_tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)

    if len(llr) != G:
        raise ValueError(f"LLR size {len(llr)} != G={G}")

    if flip_sign:
        llr = -llr

    llr_norm = np.clip(llr / llr_clip, -1.0, 1.0)
    llr_int8 = np.round(llr_norm * 127).astype(np.int8)
    llr_int8.tofile(out_path)
    return llr_int8


# ----------------------------------------------------
# 載入 txbits (packed → bits)
# ----------------------------------------------------
def load_txbits_unpacked(bundle_dir, A):
    tx_path = os.path.join(bundle_dir, "txbits.bin")
    tx_bytes = np.fromfile(tx_path, dtype=np.uint8)
    bits = np.unpackbits(tx_bytes)[:A].astype(np.uint8)
    return bits


# ----------------------------------------------------
# 解析 ldpc_cfg.txt（OAI 產生的 txt 版本）
# ----------------------------------------------------
def parse_ldpc_cfg_txt(path):
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            key, val = parts
            try:
                cfg[key] = int(val)
            except ValueError:
                pass
    return cfg


# ----------------------------------------------------
# 計算 BER
# ----------------------------------------------------
def calc_ber(tx, dec):
    N = min(len(tx), len(dec))
    if N == 0:
        return 1.0
    return np.sum(tx[:N] != dec[:N]) / N


# ----------------------------------------------------
# 主流程
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle_root",
        type=str,
        default="/home/richard93513/SpikingRx-on-OAI/spx_records/bundle"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/richard93513/SpikingRx-on-OAI/src/train/best_spikingrx_model.pth"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="/home/richard93513/openairinterface5g/cmake_targets/ran_build/build/ldpctest_spx"
    )
    args = parser.parse_args()

    device_conv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_fc   = torch.device("cpu")

    print(f"[Device] conv={device_conv},  readout={device_fc}")

    # ----------------------------------------------------
    # 建立模型（T=3, base_ch=16）
    # ----------------------------------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,          # 👈 跟訓練時一致
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
        out_bits=14400,
        T=3,
        device_conv=device_conv,
        device_fc=device_fc,
    )

    # ----------------------------------------------------
    # 載入權重
    # ----------------------------------------------------
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"找不到 checkpoint: {args.ckpt}")

    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    # ----------------------------------------------------
    # 掃描 bundles
    # ----------------------------------------------------
    bundle_dirs = sorted(glob.glob(f"{args.bundle_root}/f*_s*"))
    print(f"[INFO] Found {len(bundle_dirs)} bundles")

    all_results = []

    for bdir in bundle_dirs:
        print("\n====================================================")
        print(f"[Bundle] {bdir}")

        # -----------------------
        # 讀取 config
        # -----------------------
        cfg_txt = os.path.join(bdir, "ldpc_cfg.txt")
        if not os.path.exists(cfg_txt):
            print("[SKIP] 沒有 ldpc_cfg.txt")
            continue

        cfg = parse_ldpc_cfg_txt(cfg_txt)
        if "A" not in cfg or "G" not in cfg:
            print("[SKIP] cfg 裡沒有 A 或 G")
            continue

        A = cfg["A"]
        G = cfg["G"]

        print(f"  A={A}, G={G}")

        if G != 14400:
            print("  [SKIP] G != 14400")
            continue

        # -----------------------
        # 讀 fullgrid (T=3)
        # -----------------------
        fullgrid_path = os.path.join(bdir, "fullgrid.bin")
        if not os.path.exists(fullgrid_path):
            print("[SKIP] 沒 fullgrid.bin")
            continue

        x, fg_meta = load_oai_fullgrid(
            fullgrid_path,
            H_out=32,
            W_out=32,
            T=3,
            device=device_conv,
        )

        # -----------------------
        # Inference
        # -----------------------
        with torch.no_grad():
            llr_vec, aux = model(x)
        # ===== 原本的 DEBUG（保留）=====
        print("[DEBUG] spike_rate_per_stage =")
        print(aux["spike_rate_per_stage"])
        print("[DEBUG] final_rate_mean =", aux["final_rate_mean"])
        print("[DEBUG] final_rate_std  =", aux["final_rate_std"])

        # ===== 成果證據用（新增，重點）=====
        llr_np = llr_vec.cpu().numpy()
        print("[RESULT] LLR length =", len(llr_np))
        print("[RESULT] LLR[0:16] =", llr_np[:16])
        # ==================================

        llr_float_path = os.path.join(bdir, "infer_llr_float.npy")
        np.save(llr_float_path, llr_vec.cpu().numpy())
        print("  → Saved float LLR for debug:", llr_float_path)

        # -----------------------
        # Save LLR (int8)
        # -----------------------
        llr_path = os.path.join(bdir, "infer_llr_int8.bin")
        save_llr_for_oai_decoder(llr_vec, llr_path, llr_clip=8.0, flip_sign=True, G=G)
        print(f"  → Saved LLR: {llr_path}")
	
        # -----------------------
        # 呼叫 OAI ldpctest_spx
        # -----------------------
        dec_path = os.path.join(bdir, "decoded_bits.bin")
        cmd = [
            args.decoder,
            llr_path,
            cfg_txt,
            dec_path,
        ]
        print("  exec:", " ".join(cmd))
        subprocess.run(cmd)

        if not os.path.exists(dec_path):
            print("  [FAIL] decoder 沒輸出 decoded_bits.bin")
            continue

        # -----------------------
        # 計算 BER
        # -----------------------
        tx_bits = load_txbits_unpacked(bdir, A)
        dec_bits = np.fromfile(dec_path, dtype=np.uint8)[:A]

        ber = calc_ber(tx_bits, dec_bits)
        print(f"  [BER] {ber:.6f}")

        all_results.append((os.path.basename(bdir), ber))

    # ----------------------------------------------------
    # 最後總結
    # ----------------------------------------------------
    print("\n==================== ALL RESULTS ====================")
    for name, ber in all_results:
        print(f"{name:15s} → BER = {ber:.6f}")


if __name__ == "__main__":
    main()
    

