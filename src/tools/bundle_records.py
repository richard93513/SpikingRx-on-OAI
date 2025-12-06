# bundle_records.py
# -----------------------------------------------------------
# 將 spx_records/raw/ 的 fullgrid、txbits、ldpc_cfg、LLR 自動配對
# 並額外產生一份 ldpc_cfg.txt（給 ldpctest_spx 用）
#
# Output:
#   spx_records/bundle/fXXXX_sYY/
#       - fullgrid.bin
#       - txbits.bin
#       - oai_llr.bin      ← ★ 新增：OAI demapper dump 的 LLR
#       - ldpc_cfg.json
#       - ldpc_cfg.txt
#       - meta.json
#
# -----------------------------------------------------------

import os
import re
import json
import shutil
from glob import glob

ROOT = os.path.expanduser("~/SpikingRx-on-OAI/spx_records")
RAW_DIR = os.path.join(ROOT, "raw")
BUNDLE_DIR = os.path.join(ROOT, "bundle")

os.makedirs(BUNDLE_DIR, exist_ok=True)

# --------------------------------------------------------
# 正規表示式
# --------------------------------------------------------

# fullgrid：fXXXX_sYY_fullgrid_idxIIIIII.bin
FULLGRID_RE = re.compile(
    r"f(?P<frame>\d+)_s(?P<slot>\d+)_fullgrid_idx(?P<idx>\d+)\.bin"
)

# txbits：fXXXX_sYY_txbits_idxIIIIII_rntiRRRRR.bin
TXBITS_RE = re.compile(
    r"f(?P<frame>\d+)_s(?P<slot>\d+)_txbits_idx(?P<idx>\d+)_rnti(?P<rnti>\d+)\.bin"
)

# ldpc json：fXXXX_sYY_ldpc.json
LDPC_RE = re.compile(
    r"f(?P<frame>\d+)_s(?P<slot>\d+)_ldpc\.json"
)

# ★ 新增：OAI demapper LLR：fXXXX_sYY_llr_idxIIIIII.bin
LLR_RE = re.compile(
    r"f(?P<frame>\d+)_s(?P<slot>\d+)_llr_idx(?P<idx>\d+)\.bin"
)

# --------------------------------------------------------
# 將 JSON 轉為 key-value txt（ldpctest_spx 使用格式）
# --------------------------------------------------------
def write_ldpc_cfg_txt(json_path, txt_path):
    with open(json_path) as f:
        cfg = json.load(f)

    # 寫入 key-value 格式
    with open(txt_path, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k} {v}\n")

    print(f"  [TXT] wrote {os.path.basename(txt_path)}")


# --------------------------------------------------------
# 掃描 RAW/ 目錄
# --------------------------------------------------------
def scan_raw_records():
    fullgrid_files = []
    txbits_files = []
    ldpc_files = []
    llr_files = []

    # 掃描 bin (fullgrid + txbits + llr)
    for f in glob(os.path.join(RAW_DIR, "*.bin")):
        base = os.path.basename(f)

        m = FULLGRID_RE.match(base)
        if m:
            info = m.groupdict()
            info["path"] = f
            fullgrid_files.append(info)
            continue

        m = TXBITS_RE.match(base)
        if m:
            info = m.groupdict()
            info["path"] = f
            txbits_files.append(info)
            continue

        m = LLR_RE.match(base)
        if m:
            info = m.groupdict()
            info["path"] = f
            llr_files.append(info)
            continue

    # 掃描 JSON (ldpc cfg)
    for j in glob(os.path.join(RAW_DIR, "*.json")):
        base = os.path.basename(j)
        m = LDPC_RE.match(base)
        if m:
            info = m.groupdict()
            info["path"] = j
            ldpc_files.append(info)

    return fullgrid_files, txbits_files, ldpc_files, llr_files


# --------------------------------------------------------
# Bundling: fullgrid + txbits + ldpc + llr 都存在才打包
# --------------------------------------------------------
def bundle_records():
    fullgrid_list, txbits_list, ldpc_list, llr_list = scan_raw_records()

    print(f"Found {len(fullgrid_list)} fullgrid")
    print(f"Found {len(txbits_list)} txbits")
    print(f"Found {len(ldpc_list)} ldpc cfg")
    print(f"Found {len(llr_list)} llr")

    # txbits 依 frame 分組（跟以前一樣）
    txbits_by_frame = {}
    for t in txbits_list:
        frame = int(t["frame"])
        txbits_by_frame.setdefault(frame, []).append(t)

    # ldpc cfg 用 (frame, slot) 索引
    ldpc_by_fs = {(int(c["frame"]), int(c["slot"])): c for c in ldpc_list}

    # ★ llr 也用 (frame, slot) 分組，再根據 idx 找最近
    llr_by_fs = {}
    for l in llr_list:
        frame = int(l["frame"])
        slot = int(l["slot"])
        llr_by_fs.setdefault((frame, slot), []).append(l)

    total_bundled = 0

    for fg in fullgrid_list:
        frame = int(fg["frame"])
        slot = int(fg["slot"])
        fg_idx = int(fg["idx"])

        # 1. fullgrid 有了 → 找 txbits（同一 frame 中 idx 最接近）
        if frame not in txbits_by_frame:
            print(f"[SKIP] frame {frame} slot {slot}: missing txbits → skip")
            continue

        candidates_tx = txbits_by_frame[frame]
        best_tx = min(candidates_tx, key=lambda x: abs(int(x["idx"]) - fg_idx))

        # 2. 找 LDPC cfg（frame + slot）
        ldpc_key = (frame, slot)
        if ldpc_key not in ldpc_by_fs:
            print(f"[SKIP] frame {frame} slot {slot}: missing ldpc_cfg → skip")
            continue

        ldpc_info = ldpc_by_fs[ldpc_key]

        # 3. 找 LLR（frame + slot + idx 最接近）
        if ldpc_key not in llr_by_fs:
            print(f"[SKIP] frame {frame} slot {slot}: missing LLR → skip")
            continue

        candidates_llr = llr_by_fs[ldpc_key]
        best_llr = min(candidates_llr, key=lambda x: abs(int(x["idx"]) - fg_idx))

        # 🎯 四件套齊全 -> 才 bundle
        out_dir = os.path.join(BUNDLE_DIR, f"f{frame:04d}_s{slot:02d}")
        os.makedirs(out_dir, exist_ok=True)

        fg_out = os.path.join(out_dir, "fullgrid.bin")
        tx_out = os.path.join(out_dir, "txbits.bin")
        ldpc_json_out = os.path.join(out_dir, "ldpc_cfg.json")
        ldpc_txt_out = os.path.join(out_dir, "ldpc_cfg.txt")
        llr_out = os.path.join(out_dir, "oai_llr.bin")  # ★ 新增：標準化檔名
        meta_out = os.path.join(out_dir, "meta.json")

        # 複製檔案
        shutil.copy2(fg["path"], fg_out)
        shutil.copy2(best_tx["path"], tx_out)
        shutil.copy2(ldpc_info["path"], ldpc_json_out)
        shutil.copy2(best_llr["path"], llr_out)

        # 產生 ldpc_cfg.txt
        write_ldpc_cfg_txt(ldpc_json_out, ldpc_txt_out)

        # 建 meta.json
        meta = {
            "frame": frame,
            "slot": slot,
            "fg_idx": fg_idx,
            "tx_idx": int(best_tx["idx"]),
            "llr_idx": int(best_llr["idx"]),
            "rnti": int(best_tx["rnti"]),
            "fullgrid_file": os.path.basename(fg["path"]),
            "txbits_file": os.path.basename(best_tx["path"]),
            "llr_file": os.path.basename(best_llr["path"]),
            "ldpc_cfg_file": os.path.basename(ldpc_info["path"]),
        }

        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] bundled f{frame:04d}_s{slot:02d}")
        total_bundled += 1

    print(f"\nDone. Total valid bundles: {total_bundled}")


if __name__ == "__main__":
    bundle_records()


