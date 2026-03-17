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
        x     : torch.FloatTensor, shape [T, 4, 14, 1272]  (default T=1)
                channel 0 = real
                channel 1 = imag
                channel 2 = dmrs_mask
                channel 3 = data_mask
        y_llr : torch.FloatTensor, shape [G]  (currently G=14400)
        cfg   : dict parsed from ldpc_cfg.txt
        bdir  : bundle directory path
    """

    def __init__(
        self,
        bundle_root,
        limit=None,
        normalize=True,
        eps=1e-6,
        T=1,
        keep_rect=True,
    ):
        super().__init__()
        self.bundle_root = bundle_root
        self.normalize = normalize
        self.eps = eps
        self.T = T
        self.keep_rect = keep_rect

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

        cfg_path = os.path.join(bdir, "ldpc_cfg.txt")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"[ERROR] ldpc_cfg.txt missing: {bdir}")

        cfg = self._parse_kv_file(cfg_path)
        if "G" not in cfg:
            raise RuntimeError(f"[ERROR] cfg missing G: {cfg_path}")

        G = int(cfg["G"])

        llr_path = os.path.join(bdir, "demapper_llr_f32.bin")
        if not os.path.exists(llr_path):
            raise FileNotFoundError(f"[ERROR] demapper_llr_f32.bin missing: {bdir}")

        llr = np.fromfile(llr_path, dtype=np.float32)
        if llr.size != G:
            raise RuntimeError(
                f"[ERROR] LLR length mismatch in {bdir}\n"
                f"G={G}, but file={llr.size}"
            )

        y_llr = torch.from_numpy(llr.astype(np.float32))

        fg_path = os.path.join(bdir, "fullgrid.bin")
        if not os.path.exists(fg_path):
            raise FileNotFoundError(f"[ERROR] fullgrid.bin missing: {bdir}")

        x_full, meta = load_oai_fullgrid(
            fg_path,
            T=self.T,
            device="cpu",
            keep_rect=self.keep_rect,
        )

        x = x_full.squeeze(0).contiguous()  # [T, 2, H, W]

        H = int(meta["n_sym"])
        W = int(meta["used_sc"])
        expected_shape = (self.T, 2, H, W)
        if x.shape != expected_shape:
            raise RuntimeError(
                f"[ERROR] unexpected x shape in {bdir}: got {tuple(x.shape)}, "
                f"expected {expected_shape}"
            )

        pdsch_cfg_path = os.path.join(bdir, "pdsch_cfg.txt")
        if not os.path.exists(pdsch_cfg_path):
            raise FileNotFoundError(f"[ERROR] pdsch_cfg.txt missing: {bdir}")

        pdsch_cfg = self._parse_kv_file(pdsch_cfg_path)

        dmrs_mask, data_mask = self._build_dmrs_and_data_mask(
            pdsch_cfg=pdsch_cfg,
            meta=meta,
            bdir=bdir,
        )  # [H, W], [H, W]

        dmrs_mask = dmrs_mask.unsqueeze(0).repeat(self.T, 1, 1)  # [T, H, W]
        data_mask = data_mask.unsqueeze(0).repeat(self.T, 1, 1)  # [T, H, W]

        x_ri = x
        if self.normalize:
            x_mean = x_ri.mean()
            x_std = x_ri.std()

            if not torch.isfinite(x_mean):
                raise RuntimeError(f"[ERROR] x mean is non-finite in {bdir}")
            if not torch.isfinite(x_std):
                raise RuntimeError(f"[ERROR] x std is non-finite in {bdir}")

            x_ri = (x_ri - x_mean) / (x_std + self.eps)

            if not torch.isfinite(x_ri).all():
                raise RuntimeError(
                    f"[ERROR] normalized x contains non-finite values in {bdir}"
                )

        x_out = torch.cat(
            [
                x_ri,
                dmrs_mask.unsqueeze(1),  # [T, 1, H, W]
                data_mask.unsqueeze(1),  # [T, 1, H, W]
            ],
            dim=1,
        )  # [T, 4, H, W]

        expected_out_shape = (self.T, 4, H, W)
        if x_out.shape != expected_out_shape:
            raise RuntimeError(
                f"[ERROR] unexpected output x shape in {bdir}: got {tuple(x_out.shape)}, "
                f"expected {expected_out_shape}"
            )

        return x_out, y_llr, cfg, bdir

    def _parse_kv_file(self, path):
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

    def _build_dmrs_and_data_mask(self, pdsch_cfg, meta, bdir):
        """
        Build first-version DMRS / DATA masks on the rectangular used-grid.

        Current assumptions are based on the user's present OAI dump setting:
            - fullgrid used region is centered 106 RB = 1272 SC
            - actual PDSCH sits in the tail RB region of the used-grid
            - for the current data collection:
                  dmrsConfigType = 0   (Type 1)
                  n_dmrs_cdm_groups = 1
                  dlDmrsSymbPos = 2052 -> symbols 2 and 11
                  start_symbol = 1
                  number_symbols = 13
                  number_rbs = 50
            - empirically, PDSCH occupies the last 21 RBs of the used-grid:
                  local RB 85..105  -> local SC 1020..1271
        """

        H = int(meta["n_sym"])
        W = int(meta["used_sc"])

        required_keys = [
            "dlDmrsSymbPos",
            "dmrsConfigType",
            "n_dmrs_cdm_groups",
            "start_symbol",
            "number_symbols",
            "number_rbs",
        ]
        missing = [k for k in required_keys if k not in pdsch_cfg]
        if missing:
            raise RuntimeError(
                f"[ERROR] pdsch_cfg missing keys {missing} in {bdir}"
            )

        dlDmrsSymbPos = int(pdsch_cfg["dlDmrsSymbPos"])
        dmrsConfigType = int(pdsch_cfg["dmrsConfigType"])
        n_dmrs_cdm_groups = int(pdsch_cfg["n_dmrs_cdm_groups"])
        start_symbol = int(pdsch_cfg["start_symbol"])
        number_symbols = int(pdsch_cfg["number_symbols"])
        number_rbs = int(pdsch_cfg["number_rbs"])

        dmrs_symbols = [s for s in range(H) if ((dlDmrsSymbPos >> s) & 0x1) != 0]

        if dmrsConfigType == 0 and n_dmrs_cdm_groups == 1:
            dmrs_re = {0, 2, 4, 6, 8, 10}
        elif dmrsConfigType == 1 and n_dmrs_cdm_groups == 1:
            dmrs_re = {0, 1, 6, 7}
        elif dmrsConfigType == 1 and n_dmrs_cdm_groups == 2:
            dmrs_re = {0, 1, 2, 3, 6, 7, 8, 9}
        else:
            raise RuntimeError(
                f"[ERROR] unsupported DMRS config in {bdir}: "
                f"dmrsConfigType={dmrsConfigType}, "
                f"n_dmrs_cdm_groups={n_dmrs_cdm_groups}"
            )

        # Current dataset-specific placement:
        # used-grid has 106 RB, and the active PDSCH is observed at local RB 85..105.
        local_total_rbs = W // 12
        if W % 12 != 0:
            raise RuntimeError(f"[ERROR] used_sc={W} is not divisible by 12 in {bdir}")

        if local_total_rbs != 106:
            raise RuntimeError(
                f"[ERROR] unexpected local_total_rbs={local_total_rbs} in {bdir}; "
                f"current mask builder expects 106"
            )

        # From actual dump inspection:
        # active region occupies the final 21 RBs of the 106-RB used-grid.
        # This matches local RB 85..105 -> SC 1020..1271.
        local_start_rb = 0
        local_number_rbs = number_rbs

        
        dmrs_mask = torch.zeros((H, W), dtype=torch.float32)
        data_mask = torch.zeros((H, W), dtype=torch.float32)

        sym_start = start_symbol
        sym_end = start_symbol + number_symbols

        if not (0 <= sym_start < H and 0 < sym_end <= H):
            raise RuntimeError(
                f"[ERROR] invalid symbol range in {bdir}: "
                f"start_symbol={start_symbol}, number_symbols={number_symbols}, H={H}"
            )

        for s in range(sym_start, sym_end):
            is_dmrs_symbol = s in dmrs_symbols

            for rb in range(local_start_rb, local_start_rb + local_number_rbs):
                rb_base = rb * 12
                for re in range(12):
                    sc = rb_base + re
                    if sc < 0 or sc >= W:
                        continue

                    if is_dmrs_symbol and re in dmrs_re:
                        dmrs_mask[s, sc] = 1.0
                    else:
                        data_mask[s, sc] = 1.0

        return dmrs_mask, data_mask
