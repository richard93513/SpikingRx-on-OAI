#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bundle_records.py (mtime-based pairing)

將 spx_records/raw/ 的：
  - fullgrid (bin)
  - txbits  (bin)
  - llr     (bin, float32)
  - ldpc_cfg.json

依照 (frame, slot) + 檔案修改時間 mtime 的「最近鄰」進行配對。

Output:
  spx_records/bundle/fXXXX_sYY/
      - fullgrid.bin
      - txbits.bin
      - oai_llr.bin
      - ldpc_cfg.json
      - ldpc_cfg.txt
      - meta.json
"""

import os
import re
import json
import shutil
from glob import glob

ROOT = os.path.expanduser("~/SpikingRx-on-OAI/spx_records")
RAW_DIR = os.path.join(ROOT, "raw")
BUNDLE_DIR = os.path.join(ROOT, "bundle")
os.makedirs(BUNDLE_DIR, exist_ok=True)

FULLGRID_RE = re.compile(r"f(?P<frame>\d+)_s(?P<slot>\d+)_fullgrid_idx(?P<idx>\d+)\.bin")
TXBITS_RE   = re.compile(r"f(?P<frame>\d+)_s(?P<slot>\d+)_txbits_idx(?P<idx>\d+)_rnti(?P<rnti>\d+)\.bin")
LLR_RE      = re.compile(r"f(?P<frame>\d+)_s(?P<slot>\d+)_llr_idx(?P<idx>\d+)\.bin")
LDPC_RE     = re.compile(r"f(?P<frame>\d+)_s(?P<slot>\d+)_ldpc\.json")


def _mtime(p: str) -> float:
    return os.path.getmtime(p)


def write_ldpc_cfg_txt(json_path, txt_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)
    with open(txt_path, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k} {v}\n")


def scan_raw_records():
    fullgrid, txbits, llr, ldpc = [], [], [], []

    for p in glob(os.path.join(RAW_DIR, "*.bin")):
        b = os.path.basename(p)

        m = FULLGRID_RE.match(b)
        if m:
            d = m.groupdict()
            d["path"] = p
            d["mtime"] = _mtime(p)
            fullgrid.append(d)
            continue

        m = TXBITS_RE.match(b)
        if m:
            d = m.groupdict()
            d["path"] = p
            d["mtime"] = _mtime(p)
            txbits.append(d)
            continue

        m = LLR_RE.match(b)
        if m:
            d = m.groupdict()
            d["path"] = p
            d["mtime"] = _mtime(p)
            llr.append(d)
            continue

    for p in glob(os.path.join(RAW_DIR, "*.json")):
        b = os.path.basename(p)
        m = LDPC_RE.match(b)
        if m:
            d = m.groupdict()
            d["path"] = p
            d["mtime"] = _mtime(p)
            ldpc.append(d)

    return fullgrid, txbits, llr, ldpc


def pick_nearest_by_mtime(candidates, target_mtime):
    # candidates: list of dict with ["mtime"]
    return min(candidates, key=lambda x: abs(float(x["mtime"]) - float(target_mtime)))


def bundle_records():
    fullgrid_list, txbits_list, llr_list, ldpc_list = scan_raw_records()

    print(f"Found {len(fullgrid_list)} fullgrid")
    print(f"Found {len(txbits_list)} txbits")
    print(f"Found {len(llr_list)} llr")
    print(f"Found {len(ldpc_list)} ldpc cfg")

    # group by (frame, slot)
    tx_by_fs = {}
    for t in txbits_list:
        key = (int(t["frame"]), int(t["slot"]))
        tx_by_fs.setdefault(key, []).append(t)

    llr_by_fs = {}
    for l in llr_list:
        key = (int(l["frame"]), int(l["slot"]))
        llr_by_fs.setdefault(key, []).append(l)

    ldpc_by_fs = {}
    for c in ldpc_list:
        key = (int(c["frame"]), int(c["slot"]))
        # 若同一 fs 有多個 ldpc json，取 mtime 最新（通常較合理）
        if key not in ldpc_by_fs or float(c["mtime"]) > float(ldpc_by_fs[key]["mtime"]):
            ldpc_by_fs[key] = c

    total = 0

    for fg in fullgrid_list:
        frame = int(fg["frame"])
        slot  = int(fg["slot"])
        fg_idx = int(fg["idx"])
        fg_mtime = float(fg["mtime"])

        fs_key = (frame, slot)

        # 1) ldpc cfg 必須存在
        if fs_key not in ldpc_by_fs:
            print(f"[SKIP] f{frame} s{slot}: missing ldpc_cfg")
            continue
        ldpc_info = ldpc_by_fs[fs_key]

        # 2) llr 必須存在
        if fs_key not in llr_by_fs or len(llr_by_fs[fs_key]) == 0:
            print(f"[SKIP] f{frame} s{slot}: missing llr")
            continue
        best_llr = pick_nearest_by_mtime(llr_by_fs[fs_key], fg_mtime)

        # 3) txbits 必須存在
        if fs_key not in tx_by_fs or len(tx_by_fs[fs_key]) == 0:
            print(f"[SKIP] f{frame} s{slot}: missing txbits")
            continue
        best_tx = pick_nearest_by_mtime(tx_by_fs[fs_key], fg_mtime)

        # 4) output
        out_dir = os.path.join(BUNDLE_DIR, f"f{frame:04d}_s{slot:02d}")
        os.makedirs(out_dir, exist_ok=True)

        fg_out = os.path.join(out_dir, "fullgrid.bin")
        tx_out = os.path.join(out_dir, "txbits.bin")
        llr_out = os.path.join(out_dir, "oai_llr.bin")
        ldpc_json_out = os.path.join(out_dir, "ldpc_cfg.json")
        ldpc_txt_out  = os.path.join(out_dir, "ldpc_cfg.txt")
        meta_out      = os.path.join(out_dir, "meta.json")

        shutil.copy2(fg["path"], fg_out)
        shutil.copy2(best_tx["path"], tx_out)
        shutil.copy2(best_llr["path"], llr_out)
        shutil.copy2(ldpc_info["path"], ldpc_json_out)

        write_ldpc_cfg_txt(ldpc_json_out, ldpc_txt_out)

        meta = {
            "frame": frame,
            "slot": slot,
            "fg_idx": fg_idx,
            "fg_mtime": fg_mtime,
            "tx_idx": int(best_tx["idx"]),
            "tx_mtime": float(best_tx["mtime"]),
            "llr_idx": int(best_llr["idx"]),
            "llr_mtime": float(best_llr["mtime"]),
            "rnti": int(best_tx["rnti"]),
            "fullgrid_file": os.path.basename(fg["path"]),
            "txbits_file": os.path.basename(best_tx["path"]),
            "llr_file": os.path.basename(best_llr["path"]),
            "ldpc_cfg_file": os.path.basename(ldpc_info["path"]),
        }
        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] bundled f{frame:04d}_s{slot:02d} (fg_idx={fg_idx}, tx_idx={meta['tx_idx']}, llr_idx={meta['llr_idx']})")
        total += 1

    print(f"\nDone. Total valid bundles: {total}")


if __name__ == "__main__":
    bundle_records()



