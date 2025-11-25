# src/data/oai_to_spikingrx_tensor.py
# -*- coding: utf-8 -*-
# -------------------------------------------------------------
#  SpikingRx-on-OAI 專用 full-grid loader
#  對應 C 端產生的
#    /tmp/spx_fullgrid_f<frame>_s<slot>.bin
#
#  檔案內容格式：
#    [Header] 8 個 uint16 (frame, slot, start_symbol, n_sym, n_sc, rx_ant, cw, reserved)
#    [Payload] n_sym * n_sc 個 c16_t (int16 real + int16 imag)
#
#  最終產生 SpikingRx 輸入：
#    Tensor shape = [1, T, 2, 32, 32]
# -------------------------------------------------------------

import numpy as np
import torch
import glob
import os


# -------------------------------------------------------------
#  主功能：讀取 full-grid，轉成 SpikingRx tensor
# -------------------------------------------------------------
def load_oai_fullgrid(
    path: str,
    H_out: int = 32,
    W_out: int = 32,
    T: int = 5,
    n_rb: int = 106,         # band78 => 106 PRB
    device="cpu",
):
    # 1) 讀 entire file (int16)
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size < 8:
        raise ValueError(f"File too small: {path}")

    # 2) parse header (8 * uint16)
    header = raw[:8].view(np.uint16)
    frame        = int(header[0])
    slot         = int(header[1])
    start_symbol = int(header[2])
    n_sym        = int(header[3])
    n_sc         = int(header[4])
    rx_ant       = int(header[5])
    cw           = int(header[6])

    # 3) parse payload
    iq_int16 = raw[8:]                               # 剩下全是 I/Q
    total_re = n_sym * n_sc
    assert iq_int16.size == 2 * total_re, \
        f"IQ count mismatch: header={total_re}, file={iq_int16.size//2}"

    iq_pair = iq_int16.reshape(-1, 2)                # [N,2] = [real, imag]
    cpx = iq_pair[:, 0].astype(np.float32) + 1j * iq_pair[:, 1].astype(np.float32)
    grid_full = cpx.reshape(n_sym, n_sc)             # [14, 2048]

    # ---------------------------------------------------------
    # 4) 擷取中間 BWP (106 PRB = 1272 SC)
    # ---------------------------------------------------------
    sc_per_rb = 12
    used_sc = n_rb * sc_per_rb                       # 1272
    assert used_sc <= n_sc

    first_sc = (n_sc - used_sc) // 2                 # 2048 - 1272 = 776 → 388
    last_sc  = first_sc + used_sc                    # 388..1659

    grid_used = grid_full[:, first_sc:last_sc]       # [14, 1272]

    H_raw = n_sym
    W_raw = used_sc                                  # 1272

    # ---------------------------------------------------------
    # 5) 頻域壓縮 (1272 → W_out=32)
    # ---------------------------------------------------------
    ratio = W_raw / W_out
    freq32 = np.zeros((H_raw, W_out), dtype=np.complex64)

    for i in range(W_out):
        s = int(i * ratio)
        e = int((i + 1) * ratio)
        if e == s:
            e = s + 1
        freq32[:, i] = grid_used[:, s:e].mean(axis=1)

    # ---------------------------------------------------------
    # 6) 時域補零 (14 → 32)
    # ---------------------------------------------------------
    out = np.zeros((H_out, W_out), dtype=np.complex64)
    out[:H_raw, :] = freq32

    # ---------------------------------------------------------
    # 7) 轉成 (C=2, H_out, W_out)
    # ---------------------------------------------------------
    chw = np.zeros((2, H_out, W_out), dtype=np.float32)
    chw[0] = out.real
    chw[1] = out.imag

    # ---------------------------------------------------------
    # 8) 堆成 (1, T, 2, 32, 32)
    # ---------------------------------------------------------
    x = np.stack([chw] * T, axis=0)       # [T,2,H,W]
    x = x[np.newaxis, ...]                # [1,T,2,H,W]

    meta = {
        "path": path,
        "frame": frame,
        "slot": slot,
        "start_symbol": start_symbol,
        "n_sym": n_sym,
        "n_sc": n_sc,
        "first_sc": first_sc,
        "used_sc": used_sc,
        "rx_ant": rx_ant,
        "cw": cw,
    }

    return torch.tensor(x, dtype=torch.float32, device=device), meta


# -------------------------------------------------------------
#  自動抓最新的 full-grid dump
# -------------------------------------------------------------
def load_latest_fullgrid(
    pattern="/tmp/spx_fullgrid_f*_s*.bin",
    **kwargs,
):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No fullgrid dump found.")

    latest = max(files, key=os.path.getmtime)
    return load_oai_fullgrid(latest, **kwargs)


# -------------------------------------------------------------
#  Debug Run
# -------------------------------------------------------------
if __name__ == "__main__":
    x, meta = load_latest_fullgrid()
    print("Loaded file:", meta["path"])
    print("Frame/Slot:", meta["frame"], meta["slot"])
    print("Tensor shape:", x.shape)
    print("Meta:", meta)


