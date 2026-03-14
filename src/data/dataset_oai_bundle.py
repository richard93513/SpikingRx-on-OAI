# src/data/dataset_oai_bundle.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data.oai_to_spikingrx_tensor import load_oai_fullgrid


class OAI_Bundle_Dataset(Dataset):
    """
    Dataset for OAI bundle training.

    Output:
        x     : torch.FloatTensor, shape [1, 2, 32, 64]
        y_llr : torch.FloatTensor, shape [G]  (currently G=14400)
        cfg   : dict parsed from ldpc_cfg.txt
        bdir  : bundle directory path

    Notes:
        1) Label is FIXED to demapper_llr_f32.bin
           We do NOT mix with oai_llr.bin anymore.

        2) Input x is normalized per-sample with z-score:
               x = (x - mean) / (std + eps)
           This reduces bundle-to-bundle gain variation.
    """

    def __init__(self, bundle_root, limit=None, normalize=True, eps=1e-6):
        super().__init__()
        self.bundle_root = bundle_root
        self.normalize = normalize
        self.eps = eps

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

        # --------------------------
        # 1) read ldpc_cfg
        # --------------------------
        cfg_path = os.path.join(bdir, "ldpc_cfg.txt")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"[ERROR] ldpc_cfg.txt missing: {bdir}")

        cfg = self._parse_ldpc_cfg(cfg_path)

        if "G" not in cfg:
            raise RuntimeError(f"[ERROR] cfg missing G: {cfg_path}")

        G = int(cfg["G"])

        # --------------------------
        # 2) read LLR label
        # FIXED: only use demapper_llr_f32.bin
        # --------------------------
        llr_path = os.path.join(bdir, "demapper_llr_f32.bin")

        if not os.path.exists(llr_path):
            raise FileNotFoundError(
                f"[ERROR] demapper_llr_f32.bin missing: {bdir}"
            )

        llr = np.fromfile(llr_path, dtype=np.float32)

        if llr.size != G:
            raise RuntimeError(
                f"[ERROR] LLR length mismatch in {bdir}\n"
                f"G={G}, but file={llr.size}"
            )

        y_llr = torch.from_numpy(llr.astype(np.float32))

        # --------------------------
        # 3) read fullgrid
        # --------------------------
        fg_path = os.path.join(bdir, "fullgrid.bin")

        if not os.path.exists(fg_path):
            raise FileNotFoundError(f"[ERROR] fullgrid.bin missing: {bdir}")

        x_full, _ = load_oai_fullgrid(
            fg_path,
            H_out=32,
            W_out=64,
            T=5,
            device="cpu",
        )

        # x_full expected shape: [1, T, 2, 32, 32]
        x = x_full.squeeze(0).contiguous()

        # use first 1 time steps
        x = x[:1]

        if x.shape != (1, 2, 32, 64):
            raise RuntimeError(
                f"[ERROR] unexpected x shape in {bdir}: got {tuple(x.shape)}, "
                f"expected (1, 2, 32, 64)"
            )

        # --------------------------
        # 4) per-sample normalization
        # --------------------------
        if self.normalize:
            x_mean = x.mean()
            x_std = x.std()

            if not torch.isfinite(x_mean):
                raise RuntimeError(f"[ERROR] x mean is non-finite in {bdir}")
            if not torch.isfinite(x_std):
                raise RuntimeError(f"[ERROR] x std is non-finite in {bdir}")

            x = (x - x_mean) / (x_std + self.eps)

            if not torch.isfinite(x).all():
                raise RuntimeError(f"[ERROR] normalized x contains non-finite values in {bdir}")

        return x, y_llr, cfg, bdir

    # --------------------------
    # parse ldpc_cfg.txt
    # --------------------------
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
                    p = line.split()
                    if len(p) != 2:
                        continue
                    k, v = p

                k = k.strip()
                v = v.strip()

                try:
                    v = int(v)
                except Exception:
                    pass

                cfg[k] = v

        return cfg
