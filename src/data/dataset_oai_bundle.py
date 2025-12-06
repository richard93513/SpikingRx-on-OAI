# src/data/dataset_oai_bundle.py
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from data.oai_to_spikingrx_tensor import load_oai_fullgrid


class OAI_Bundle_Dataset(Dataset):
    """
    每一筆 sample 對應一個 bundle/fXXXX_sYY/，包含：
      x         : fullgrid → tensor, shape = [T=5 → T=3, 2, 32, 32]
      y_llr     : OAI demapper LLR, shape = [G], float32
      cfg       : 從 ldpc_cfg.txt 解析出的字典（包含 A, C, F, G...）
      bundle_dir: 該 bundle 的完整路徑

    ★ 新版 dataset 不再讀 txbits
      → label = oai_llr.bin (G bits)
      → 用於 MSE(pred_LLR , true_LLR)
    """

    def __init__(self, bundle_root, limit=None):
        super().__init__()
        self.bundle_root = bundle_root

        dirs = sorted(
            d for d in os.listdir(bundle_root)
            if d.startswith("f") and os.path.isdir(os.path.join(bundle_root, d))
        )
        if limit is not None:
            dirs = dirs[:limit]

        self.bundle_dirs = [os.path.join(bundle_root, d) for d in dirs]
        print(f"[Dataset] Found {len(self.bundle_dirs)} bundles")

    def __len__(self):
        return len(self.bundle_dirs)

    def __getitem__(self, idx):
        bdir = self.bundle_dirs[idx]

        # -----------------------
        # 1) 讀 ldpc_cfg.txt → cfg dict
        # -----------------------
        cfg_path = os.path.join(bdir, "ldpc_cfg.txt")
        cfg = self._parse_ldpc_cfg(cfg_path)

        if "G" not in cfg:
            raise KeyError(f"[ERROR] cfg 沒有 G：{cfg_path}")
        G = int(cfg["G"])

        # -----------------------
        # 2) OAI demapper 的 LLR → label
        # -----------------------
        llr_path = os.path.join(bdir, "oai_llr.bin")
        if not os.path.exists(llr_path):
            raise FileNotFoundError(f"[ERROR] 找不到 oai_llr.bin：{llr_path}")

        llr = np.fromfile(llr_path, dtype=np.float32)
        if llr.size != G:
            raise ValueError(
                f"[ERROR] oai_llr.bin 長度 {llr.size} != G={G} at {bdir}"
            )

        y_llr = torch.from_numpy(llr.astype(np.float32))  # [G]

        # -----------------------
        # 3) fullgrid → tensor
        # -----------------------
        fg_path = os.path.join(bdir, "fullgrid.bin")
        if not os.path.exists(fg_path):
            raise FileNotFoundError(f"[ERROR] 找不到 fullgrid.bin：{fg_path}")

        # 回傳 [1, 5, 2, 32, 32]
        x_full, _ = load_oai_fullgrid(
            fg_path,
            H_out=32,
            W_out=32,
            T=5,
            device="cpu",
        )
        x = x_full.squeeze(0).contiguous()   # → [5, 2, 32, 32]

        # -----------------------
        # ⭐⭐⭐ 重要：只取前 3 個時序 → 模型 T=3
        # -----------------------
        x = x[:3]    # → [3, 2, 32, 32]

        return x, y_llr, cfg, bdir

    # ---------------------------
    # parse ldpc_cfg.txt
    # ---------------------------
    def _parse_ldpc_cfg(self, path):
        cfg = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "=" in line:
                    k, v = line.split("=", 1)
                else:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    k, v = parts

                k = k.strip()
                v = v.strip()

                try:
                    v = int(v)
                except ValueError:
                    pass

                cfg[k] = v

        return cfg


