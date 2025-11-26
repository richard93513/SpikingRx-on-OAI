# bundle_records.py
# -----------------------------------------------------------
# 將 spx_records/raw/ 的 fullgrid 與 txbits 自動配對
# 根據 frame + slot (RX) 建立 bundle/
#
# Output:
#   spx_records/bundle/fXXXX_sYY/
#       - fullgrid.bin
#       - txbits.bin
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
# 檔名解析器
# --------------------------------------------------------

# fullgrid: f0458_s01_fullgrid_idx000003.bin
FULLGRID_RE = re.compile(
    r"f(?P<frame>\d+)_s(?P<slot>\d+)_fullgrid_idx(?P<idx>\d+)\.bin"
)

# txbits: f0458_s00_txbits_idx001243_rnti65535.bin
TXBITS_RE = re.compile(
    r"f(?P<frame>\d+)_s(?P<slot>\d+)_txbits_idx(?P<idx>\d+)_rnti(?P<rnti>\d+)\.bin"
)


# --------------------------------------------------------
# 掃描 RAW 目錄
# --------------------------------------------------------
def scan_raw_records():
    fullgrid_files = []
    txbits_files = []

    for f in glob(os.path.join(RAW_DIR, "*.bin")):
        base = os.path.basename(f)

        m1 = FULLGRID_RE.match(base)
        if m1:
            info = m1.groupdict()
            info["path"] = f
            fullgrid_files.append(info)
            continue

        m2 = TXBITS_RE.match(base)
        if m2:
            info = m2.groupdict()
            info["path"] = f
            txbits_files.append(info)
            continue

    return fullgrid_files, txbits_files


# --------------------------------------------------------
# 主配對邏輯
# --------------------------------------------------------
def bundle_records():
    fullgrid_list, txbits_list = scan_raw_records()

    print(f"Found {len(fullgrid_list)} fullgrid files")
    print(f"Found {len(txbits_list)} txbits files")

    # 按 frame 分組 txbits
    txbits_by_frame = {}
    for t in txbits_list:
        f = int(t["frame"])
        txbits_by_frame.setdefault(f, []).append(t)

    total_bundled = 0

    for fg in fullgrid_list:
        frame = int(fg["frame"])
        slot = int(fg["slot"])
        fg_idx = int(fg["idx"])

        # 找同 frame 的 txbits
        if frame not in txbits_by_frame:
            print(f"[WARN] frame {frame} has fullgrid but no txbits")
            continue

        candidates = txbits_by_frame[frame]

        # 先依 idx 接近排序
        candidates_sorted = sorted(
            candidates,
            key=lambda x: abs(int(x["idx"]) - fg_idx)
        )

        # 選最接近的一筆
        best = candidates_sorted[0]

        out_dir = os.path.join(BUNDLE_DIR, f"f{frame:04d}_s{slot:02d}")
        os.makedirs(out_dir, exist_ok=True)

        fg_out = os.path.join(out_dir, "fullgrid.bin")
        tx_out = os.path.join(out_dir, "txbits.bin")
        meta_out = os.path.join(out_dir, "meta.json")

        # copy
        shutil.copy2(fg["path"], fg_out)
        shutil.copy2(best["path"], tx_out)

        meta = {
            "frame": frame,
            "slot": slot,
            "fg_idx": fg_idx,
            "tx_idx": int(best["idx"]),
            "rnti": int(best["rnti"]),
            "fullgrid_file": os.path.basename(fg["path"]),
            "txbits_file": os.path.basename(best["path"]),
        }

        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2)

        total_bundled += 1
        print(f"[OK] Bundled frame={frame} slot={slot} → {out_dir}")

    print(f"\nDone. Total bundled: {total_bundled}")


if __name__ == "__main__":
    bundle_records()

