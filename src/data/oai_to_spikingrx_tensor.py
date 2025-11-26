# src/data/oai_to_spikingrx_tensor.py
# -*- coding: utf-8 -*-
# -------------------------------------------------------------
#  SpikingRx-on-OAI 專用 full-grid loader（最終版）
#
#  與 UE C 端 dump 對應的格式：
#
#   uint16 header[8]:
#     header[0] = frame
#     header[1] = slot
#     header[2] = start_symbol (固定 0)
#     header[3] = n_sym        (14)
#     header[4] = first_sc     (例如 388，BWP 起點)
#     header[5] = used_sc      (例如 1272，BWP 寬度)
#     header[6] = rx_ant
#     header[7] = cw
#
#   接著 payload:
#     n_sym * ofdm_size 個 complex16 (int16 I, int16 Q)
#     例如: 14 × 2048
#
#  最終輸出:
#     x: Tensor [1, T, 2, 32, 32]
#     meta: dict（包含 frame, slot, first_sc, used_sc, n_sc_full 等）
# -------------------------------------------------------------

import numpy as np
import torch
import os
import glob


# -------------------------------------------------------------
#  核心：讀取一個 full-grid dump 檔案 → SpikingRx 輸入張量
# -------------------------------------------------------------
def load_oai_fullgrid(
    path: str,
    H_out: int = 32,
    W_out: int = 32,
    T: int = 5,
    device: str = "cpu",
):
    # -------------------------------
    # 1) 讀 raw int16
    # -------------------------------
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size < 8:
        raise ValueError(f"[SPX][Loader] File too small: {path} (int16 count={raw.size})")

    # -------------------------------
    # 2) 解析 header (8×uint16)
    # -------------------------------
    header = raw[:8].view(np.uint16)
    frame        = int(header[0])
    slot         = int(header[1])
    start_symbol = int(header[2])
    n_sym        = int(header[3])
    first_sc     = int(header[4])
    used_sc      = int(header[5])
    rx_ant       = int(header[6])
    cw           = int(header[7])

    # -------------------------------
    # 3) 解析 payload → I/Q pairs
    # -------------------------------
    iq_int16 = raw[8:]  # 之後全部都是 I/Q
    total_iq = iq_int16.size
    if total_iq % 2 != 0:
        raise ValueError(
            f"[SPX][Loader] IQ length not even: {total_iq} (file={path})"
        )

    total_re = total_iq // 2  # 每個 RE 有 I,Q 各一個 int16

    if n_sym <= 0:
        raise ValueError(f"[SPX][Loader] Invalid n_sym={n_sym} in header for {path}")

    if total_re % n_sym != 0:
        raise ValueError(
            f"[SPX][Loader] total_re({total_re}) not divisible by n_sym({n_sym})"
        )

    n_sc_full = total_re // n_sym  # e.g. 2048

    # reshape → [N,2] → complex → [n_sym, n_sc_full]
    iq_pair = iq_int16.reshape(-1, 2)
    cpx = iq_pair[:, 0].astype(np.float32) + 1j * iq_pair[:, 1].astype(np.float32)
    grid_full = cpx.reshape(n_sym, n_sc_full)  # [14, 2048]

    # -------------------------------
    # 4) 驗證 / 修正 first_sc, used_sc
    # -------------------------------
    # 理論上 C 端已保證:
    #   first_sc = (ofdm_size - used_sc)//2
    #   used_sc  = N_RB_DL*12
    # 但這裡仍然防呆檢查一下，避免意外 dump 錯。
    if used_sc <= 0 or used_sc > n_sc_full:
        # fallback：用 106 PRB (1272) 中心 BWP
        default_used_sc = 106 * 12
        used_sc = min(default_used_sc, n_sc_full)
        first_sc = (n_sc_full - used_sc) // 2
        print(
            f"[SPX][Loader][WARN] invalid used_sc in header, fallback to center BWP:"
            f" used_sc={used_sc}, first_sc={first_sc}, n_sc_full={n_sc_full}"
        )

    last_sc = first_sc + used_sc
    if first_sc < 0 or last_sc > n_sc_full:
        # 再 fallback 一次：強制以中心 BWP
        used_sc = min(106 * 12, n_sc_full)
        first_sc = (n_sc_full - used_sc) // 2
        last_sc = first_sc + used_sc
        print(
            f"[SPX][Loader][WARN] first_sc/used_sc out of range, reset:"
            f" first_sc={first_sc}, used_sc={used_sc}, n_sc_full={n_sc_full}"
        )

    # 擷取 BWP
    grid_used = grid_full[:, first_sc:last_sc]  # [n_sym, used_sc]
    H_raw = n_sym
    W_raw = used_sc

    # -------------------------------
    # 5) 頻域壓縮 W_raw → W_out
    # -------------------------------
    ratio = W_raw / float(W_out)
    freq32 = np.zeros((H_raw, W_out), dtype=np.complex64)

    for i in range(W_out):
        # 區間 [s, e)
        s = int(i * ratio)
        e = int((i + 1) * ratio)
        if e <= s:
            e = s + 1
        if s < 0:
            s = 0
        if e > W_raw:
            e = W_raw
        # 再保險一次
        if e <= s:
            # 若真的出現空區間，就直接取單點
            s = min(max(i, 0), W_raw - 1)
            e = s + 1
        freq32[:, i] = grid_used[:, s:e].mean(axis=1)

    # -------------------------------
    # 6) 時域補零 H_raw → H_out
    # -------------------------------
    out = np.zeros((H_out, W_out), dtype=np.complex64)
    copy_H = min(H_raw, H_out)
    out[:copy_H, :] = freq32[:copy_H, :]

    # -------------------------------
    # 7) 轉成 (C=2, H_out, W_out)
    # -------------------------------
    chw = np.zeros((2, H_out, W_out), dtype=np.float32)
    chw[0] = out.real
    chw[1] = out.imag

    # -------------------------------
    # 8) 堆時間軸 T → (1, T, 2, H_out, W_out)
    # -------------------------------
    x = np.stack([chw] * T, axis=0)  # [T, 2, H, W]
    x = x[np.newaxis, ...]           # [1, T, 2, H, W]

    meta = {
        "path": path,
        "frame": frame,
        "slot": slot,
        "start_symbol": start_symbol,
        "n_sym": n_sym,
        "n_sc_full": n_sc_full,
        "first_sc": first_sc,
        "used_sc": used_sc,
        "rx_ant": rx_ant,
        "cw": cw,
    }

    return torch.tensor(x, dtype=torch.float32, device=device), meta


# -------------------------------------------------------------
#  自動抓最新 fullgrid（配合你現在的檔名 & 路徑）
# -------------------------------------------------------------
def load_latest_fullgrid(
    base_dir: str = "/home/richard93513/SpikingRx-on-OAI/spx_records/raw",
    **kwargs,
):
    pattern = os.path.join(base_dir, "f*_s*_fullgrid_idx*.bin")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"[SPX][Loader] No fullgrid dump found in {base_dir}")

    latest = max(files, key=os.path.getmtime)
    return load_oai_fullgrid(latest, **kwargs)


# -------------------------------------------------------------
#  Debug: 單獨跑這個檔案時測試 loader
# -------------------------------------------------------------
if __name__ == "__main__":
    x, meta = load_latest_fullgrid()
    print("Loaded:", meta["path"])
    print("Frame/Slot:", meta["frame"], meta["slot"])
    print("Tensor shape:", x.shape)
    print("Meta:", meta)



