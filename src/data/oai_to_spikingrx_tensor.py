# src/data/oai_to_spikingrx_tensor.py
# -*- coding: utf-8 -*-

import numpy as np
import torch


def load_oai_pdsch_grid(
    path: str,
    H_raw: int = 14,       # 真實 symbols 數（舊的 rxdataF_ext 版本）
    W_raw: int = 1272,     # 真實 subcarriers 數（106 PRB × 12）
    H_out: int = 32,
    W_out: int = 32,
    T: int = 5,
    device="cpu",
):
    """
    ⚠ 這是「舊版」：給 /tmp/ue_pdsch_slot_rxdataF_ext.bin 用的。
    現在你已經改成 FULL-GRID（spx_fullgrid_f*_s*.bin），
    正式要用的是下面的 load_oai_fullgrid()。
    這個函式先保留，之後如果你要對比可以再用。
    """

    raw = np.fromfile(path, dtype=np.int16).reshape(-1, 2)
    assert raw.shape[0] == H_raw * W_raw, \
        f"Raw size mismatch: expect {H_raw * W_raw}, got {raw.shape[0]}"

    cpx = raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)
    grid = cpx.reshape(H_raw, W_raw)  # [H_raw, W_raw] = [14, 1272]

    # 2) 頻域壓縮 (1272 → 32)
    ratio = W_raw / W_out
    freq32 = np.zeros((H_raw, W_out), dtype=np.complex64)

    for i in range(W_out):
        s = int(i * ratio)
        e = int((i + 1) * ratio)
        if e == s:
            e = s + 1
        freq32[:, i] = grid[:, s:e].mean(axis=1)

    # 3) 時域補零 (H_raw → H_out)
    out = np.zeros((H_out, W_out), dtype=np.complex64)
    out[:H_raw, :] = freq32

    # 4) 轉成 (C=2, H_out, W_out)
    chw = np.zeros((2, H_out, W_out), dtype=np.float32)
    chw[0] = out.real
    chw[1] = out.imag

    # 5) 堆成 (B=1, T, C=2, H_out, W_out)
    x = np.stack([chw] * T, axis=0)   # [T, 2, H_out, W_out]
    x = x[np.newaxis, ...]            # [1, T, 2, H_out, W_out]

    return torch.tensor(x, dtype=torch.float32, device=device)


def load_oai_fullgrid(
    path: str,
    H_out: int = 32,
    W_out: int = 32,
    T: int = 5,
    n_rb: int = 106,       # 目前你是 band78 106PRB
    device="cpu",
):
    """
    讀取 OAI FULL-GRID 檔案：
      /tmp/spx_fullgrid_f<frame>_s<slot>.bin

    C 端 header 結構 (uint16_t):
      frame
      slot
      start_symbol
      n_sym         # 實際 dump 了多少個 OFDM symbols（例：13）
      n_sc          # FFT 大小（例：2048）
      rx_ant        # 目前固定 0
      cw            # 目前固定 0
      reserved

    後面就是 n_sym * n_sc 個 c16_t（int16 real + int16 imag）：
      total int16 數 = 2 * n_sym * n_sc

    這裡做的事：
      1. 讀 header + full FFT grid
      2. 把中間的 106 PRB (106*12=1272) 切出來（假設全頻 BWP）
      3. 把 1272 subcarriers 平均壓成 32（freq pooling）
      4. 時域維度補零到 H_out=32
      5. 寫成 SpikingRx 輸入格式 [B=1, T, C=2, H_out, W_out]
    """

    # ------------------------
    # 1) 讀整個檔案 (int16)
    # ------------------------
    raw_int16 = np.fromfile(path, dtype=np.int16)
    assert raw_int16.size >= 8, f"File too small to contain header: {path}"

    # header 8 個 uint16（frame, slot, start_symbol, n_sym, n_sc, rx_ant, cw, reserved）
    header_u16 = raw_int16[:8].view(np.uint16)
    frame        = int(header_u16[0])
    slot         = int(header_u16[1])
    start_symbol = int(header_u16[2])
    n_sym        = int(header_u16[3])
    n_sc         = int(header_u16[4])
    rx_ant       = int(header_u16[5])
    cw           = int(header_u16[6])
    # header_u16[7] = reserved

    # 後面全部都是 I/Q
    iq_int16 = raw_int16[8:]
    total_re = n_sym * n_sc
    assert iq_int16.size == 2 * total_re, \
        f"RE mismatch: header={total_re}, file has {iq_int16.size//2}"

    # ------------------------
    # 2) reshape 成 (n_sym, n_sc) full FFT grid
    # ------------------------
    iq_2 = iq_int16.reshape(-1, 2)
    cpx = iq_2[:, 0].astype(np.float32) + 1j * iq_2[:, 1].astype(np.float32)
    grid_full = cpx.reshape(n_sym, n_sc)   # [n_sym, n_sc] e.g. [13, 2048]

    # ------------------------
    # 3) 截出中間 106 PRB (1272 RE)
    # ------------------------
    sc_per_rb = 12
    used_sc = n_rb * sc_per_rb           # 106 * 12 = 1272
    assert used_sc <= n_sc, "used_sc 超過 FFT 大小，n_rb 設錯了？"

    # 假設 BWP 在中間：first = (n_sc - used_sc) // 2
    first_sc = (n_sc - used_sc) // 2     # 2048 - 1272 = 776 → 388
    last_sc = first_sc + used_sc         # 388..1659

    assert first_sc >= 0 and last_sc <= n_sc, \
        f"Center RB range overflow: first={first_sc}, last={last_sc}, n_sc={n_sc}"

    grid_used = grid_full[:, first_sc:last_sc]  # [n_sym, 1272]

    H_raw = n_sym
    W_raw = used_sc

    # ------------------------
    # 4) 頻域壓縮 (1272 → W_out=32)
    # ------------------------
    ratio = W_raw / W_out
    freq32 = np.zeros((H_raw, W_out), dtype=np.complex64)

    for i in range(W_out):
        s = int(i * ratio)
        e = int((i + 1) * ratio)
        if e == s:
            e = s + 1
        freq32[:, i] = grid_used[:, s:e].mean(axis=1)

    # ------------------------
    # 5) 時域補零 (H_raw → H_out)
    # ------------------------
    out = np.zeros((H_out, W_out), dtype=np.complex64)
    out[:H_raw, :] = freq32

    # ------------------------
    # 6) 轉成 (C=2, H_out, W_out)
    # ------------------------
    chw = np.zeros((2, H_out, W_out), dtype=np.float32)
    chw[0] = out.real
    chw[1] = out.imag

    # ------------------------
    # 7) 堆成 (B=1, T, C=2, H_out, W_out)
    # ------------------------
    x = np.stack([chw] * T, axis=0)   # [T, 2, H_out, W_out]
    x = x[np.newaxis, ...]            # [1, T, 2, H_out, W_out]

    meta = {
        "frame": frame,
        "slot": slot,
        "start_symbol": start_symbol,
        "n_sym": n_sym,
        "n_sc": n_sc,
        "rx_ant": rx_ant,
        "cw": cw,
        "H_raw": H_raw,         # 真實 dump symbols 數（例：13）
        "W_raw": W_raw,         # 真實有效 subcarriers 數（1272）
        "first_sc": first_sc,   # full FFT 中心 BWP 的起點 index
        "used_sc": used_sc,
    }

    return torch.tensor(x, dtype=torch.float32, device=device), meta

