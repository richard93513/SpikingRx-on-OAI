#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_pdsch_unique.py
---------------------------------------
分析 spx_records/raw/ 裡所有 fullgrid/txbits/ldpc 記錄，
判斷每筆 PDSCH 是否不同，並輸出統計表。

放置位置: tools/check_pdsch_unique.py
執行方式:
    python tools/check_pdsch_unique.py
"""

import os
import json
import glob
import hashlib
from pathlib import Path

ROOT = os.path.expanduser("~/SpikingRx-on-OAI/spx_records/raw")


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]  # 縮短顯示


def parse_filename(fname):
    """
    解析檔名格式:
    f0404_s00_txbits_idx002249_rnti65535.bin
    f0312_s10_fullgrid_idx000004.bin
    f0424_s05_ldpc.json
    """
    base = os.path.basename(fname)
    parts = base.split("_")

    frame = int(parts[0][1:])
    slot = int(parts[1][1:])
    return frame, slot


def load_ldpc_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def main():
    print("🔍 掃描資料夾:", ROOT)
    print()

    tx_files = sorted(glob.glob(f"{ROOT}/f*_txbits_idx*.bin"))
    ldpc_files = sorted(glob.glob(f"{ROOT}/f*_ldpc.json"))
    fg_files = sorted(glob.glob(f"{ROOT}/f*_fullgrid_idx*.bin"))

    print(f"找到 TX bits: {len(tx_files)} 筆")
    print(f"找到 LDPC json: {len(ldpc_files)} 筆")
    print(f"找到 Fullgrid: {len(fg_files)} 筆")
    print()

    print("📌 分析每筆 TX bits 是否不同 ...")
    print()

    records = []

    for tx in tx_files:
        frame, slot = parse_filename(tx)
        tx_sha = sha256_of_file(tx)
        tx_size = os.path.getsize(tx)

        # 找對應 LDPC
        ldpc_pattern = f"{ROOT}/f{frame:04d}_s{slot:02d}_ldpc.json"
        ldpc = ldpc_pattern if os.path.exists(ldpc_pattern) else None
        ldpc_cfg = load_ldpc_json(ldpc) if ldpc else None

        # 找對應 fullgrid
        fg = f"{ROOT}/f{frame:04d}_s{slot:02d}_fullgrid_idx"*0  # 不用 idx，raw 不會對上
        # （raw fullgrid 只能看 frame-slot）

        records.append({
            "frame": frame,
            "slot": slot,
            "tx_size": tx_size,
            "tx_sha": tx_sha,
            "ldpc": ldpc_cfg
        })

    # 顯示表格
    print("=== PDSCH Summary ===")
    print("frame  slot  TBsize(bytes)  SHA256-prefix   BG  Zc   A")
    print("--------------------------------------------------------")

    for r in records:
        bg = r["ldpc"]["BG"] if r["ldpc"] else "-"
        zc = r["ldpc"]["Zc"] if r["ldpc"] else "-"
        A  = r["ldpc"]["A"]  if r["ldpc"] else "-"

        print(f"{r['frame']:4d}   {r['slot']:2d}     {r['tx_size']:5d}      {r['tx_sha']}   {bg}  {zc}  {A}")

    print()
    print("🟩 分析完成：若 TB size、SHA、LDPC 參數任一不同，即代表不同 PDSCH。")
    print("🟦 若需要檢查重複 TB（例如 scheduler 重送），SHA 會相同。")


if __name__ == "__main__":
    main()

