#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_llr_oai_vs_spx.py

用途：
對同一個 bundle 內的：
  - oai_llr.bin (float32, len=G)
  - infer_llr_float.npy (float32, shape [1,G] 或 [G])

做快速語意對齊驗證：
1) corr(spx, oai) 與 corr(-spx, oai)（判斷 sign）
2) std / abs-mean / 小幅度比例（判斷 scale 是否太小）
3) QPSK bit-plane swap：corr(swap(spx), oai)（判斷 Qm=2 的 bit-plane 次序）

額外提供（很便宜但很有用）：
- MSE / MAE
- 可選：小範圍 circular shift 掃描，找最佳 correlation（用來抓 buffer 起點錯）

使用方式（例）：
python3 src/tests/compare_llr_oai_vs_spx.py \
  --bundle_dir /home/richard93513/SpikingRx-on-OAI/spx_records/bundle/f0232_s11 \
  --G 14400 \
  --Qm 2 \
  --shift_scan

回傳結果你只要貼：
- corr(spx, oai)
- corr(-spx, oai)
- corr(spx_swap, oai)
以及 std 統計
我就能判斷下一步鎖定 sign / scale / order。
"""

import argparse
from pathlib import Path
import numpy as np


def load_oai_llr(oai_path: Path, G: int) -> np.ndarray:
    if not oai_path.exists():
        raise FileNotFoundError(f"找不到 oai_llr.bin: {oai_path}")
    oai = np.fromfile(str(oai_path), dtype=np.float32)
    if len(oai) != G:
        raise ValueError(f"oai_llr length {len(oai)} != G={G}")
    return oai


def load_spx_llr(npy_path: Path, G: int) -> np.ndarray:
    if not npy_path.exists():
        raise FileNotFoundError(f"找不到 infer_llr_float.npy: {npy_path}")
    spx = np.load(str(npy_path))
    spx = np.asarray(spx, dtype=np.float32).reshape(-1)
    if len(spx) != G:
        raise ValueError(f"spx_llr length {len(spx)} != G={G}")
    return spx


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    # 防止全常數造成 NaN
    sa = float(a.std())
    sb = float(b.std())
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def basic_stats(name: str, x: np.ndarray) -> None:
    absx = np.abs(x)
    print(f"[STATS] {name}:")
    print(f"  mean      = {float(x.mean()):.6e}")
    print(f"  std       = {float(x.std()):.6e}")
    print(f"  min/max   = {float(x.min()):.6e} / {float(x.max()):.6e}")
    print(f"  mean|x|   = {float(absx.mean()):.6e}")
    print(f"  p(|x|<0.5)= {float(np.mean(absx < 0.5)):.6f}")
    print(f"  p(|x|<1.0)= {float(np.mean(absx < 1.0)):.6f}")
    print("")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def qpsk_bitplane_swap(x: np.ndarray, Qm: int) -> np.ndarray:
    """
    若 Qm=2（QPSK），常見錯誤是 bit-plane 次序顛倒：
    [b0,b1,b0,b1,...] <-> [b1,b0,b1,b0,...]
    """
    if Qm != 2:
        raise ValueError("bitplane swap 只在 Qm=2（QPSK）時有意義")
    return x.reshape(-1, 2)[:, ::-1].reshape(-1)


def shift_scan_best_corr(spx: np.ndarray, oai: np.ndarray, shifts: list[int]) -> tuple[int, float]:
    best_s = 0
    best_c = -1.0
    for s in shifts:
        spx_s = np.roll(spx, s)
        c = corrcoef(spx_s, oai)
        if np.isnan(c):
            continue
        if c > best_c:
            best_c = c
            best_s = s
    return best_s, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", type=str, required=True, help="bundle 路徑，如 .../bundle/f0232_s11")
    ap.add_argument("--G", type=int, default=14400, help="LLR 長度 G（預設 14400）")
    ap.add_argument("--Qm", type=int, default=2, help="調變 bits/symbol（QPSK=2）")
    ap.add_argument("--oai_name", type=str, default="oai_llr.bin", help="OAI LLR 檔名（預設 oai_llr.bin）")
    ap.add_argument("--spx_name", type=str, default="infer_llr_float.npy", help="SpikingRx LLR 檔名（預設 infer_llr_float.npy）")
    ap.add_argument("--shift_scan", action="store_true", help="啟用 small shift scan（用於偵測起點/對齊偏移）")
    args = ap.parse_args()

    bdir = Path(args.bundle_dir)
    if not bdir.exists():
        raise FileNotFoundError(f"bundle_dir 不存在: {bdir}")

    oai_path = bdir / args.oai_name
    spx_path = bdir / args.spx_name

    G = int(args.G)
    Qm = int(args.Qm)

    oai = load_oai_llr(oai_path, G)
    spx = load_spx_llr(spx_path, G)

    print("====================================================")
    print(f"[BUNDLE] {bdir}")
    print(f"[FILES] oai={oai_path.name}, spx={spx_path.name}")
    print(f"[PARAM] G={G}, Qm={Qm}")
    print("====================================================\n")

    # 1) 基本統計（scale 判斷）
    basic_stats("OAI_LLR", oai)
    basic_stats("SPX_LLR", spx)

    # 2) correlation：原始 / sign flip
    c0 = corrcoef(spx, oai)
    c1 = corrcoef(-spx, oai)

    print("[CORR] corr(spx, oai)   =", f"{c0:.6f}")
    print("[CORR] corr(-spx, oai)  =", f"{c1:.6f}")
    print("")

    # 3) MSE / MAE（額外量化對齊程度）
    print("[ERR ] MSE(spx, oai)    =", f"{mse(spx, oai):.6e}")
    print("[ERR ] MAE(spx, oai)    =", f"{mae(spx, oai):.6e}")
    print("")

    # 4) QPSK bit-plane swap（只對 Qm=2）
    if Qm == 2:
        spx_swap = qpsk_bitplane_swap(spx, Qm=2)
        c_swap = corrcoef(spx_swap, oai)
        c_swap_neg = corrcoef(-spx_swap, oai)
        print("[SWAP] corr(spx_swap, oai)  =", f"{c_swap:.6f}")
        print("[SWAP] corr(-spx_swap, oai) =", f"{c_swap_neg:.6f}")
        print("")
    else:
        print("[SWAP] Qm != 2，略過 bit-plane swap 檢查\n")

    # 5) 可選：shift scan（抓起點偏移）
    if args.shift_scan:
        shifts = list(range(-256, 257, 8))  # 小範圍粗掃；必要時可再縮步長
        best_s, best_c = shift_scan_best_corr(spx, oai, shifts)
        best_s_neg, best_c_neg = shift_scan_best_corr(-spx, oai, shifts)
        print("[SHIFT_SCAN] shifts =", f"{shifts[0]}..{shifts[-1]} step 8")
        print("[SHIFT_SCAN] best shift for spx     =", best_s, "best corr =", f"{best_c:.6f}")
        print("[SHIFT_SCAN] best shift for (-spx)  =", best_s_neg, "best corr =", f"{best_c_neg:.6f}")
        print("")

        if Qm == 2:
            spx_swap = qpsk_bitplane_swap(spx, Qm=2)
            best_s_sw, best_c_sw = shift_scan_best_corr(spx_swap, oai, shifts)
            best_s_sw_neg, best_c_sw_neg = shift_scan_best_corr(-spx_swap, oai, shifts)
            print("[SHIFT_SCAN][SWAP] best shift for spx_swap    =", best_s_sw, "best corr =", f"{best_c_sw:.6f}")
            print("[SHIFT_SCAN][SWAP] best shift for (-spx_swap) =", best_s_sw_neg, "best corr =", f"{best_c_sw_neg:.6f}")
            print("")

    print("Done.")


if __name__ == "__main__":
    main()

