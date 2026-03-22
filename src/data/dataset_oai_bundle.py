# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

from data.oai_to_spikingrx_tensor import load_oai_fullgrid


class OAI_Bundle_Dataset(Dataset):
    """
    Dataset for SpikingRx-on-OAI serial-pipeline training.

    Returns
    -------
    x_out : torch.FloatTensor
        Shape [T, 4, H, W]
            channel 0 = real
            channel 1 = imag
            channel 2 = dmrs_mask
            channel 3 = data_mask

    target : dict
        target["y_llr"]      : [G]
        target["y_eq"]       : [2, H, W]
        target["y_ch"]       : [2, H, W]
        target["eq_mask"]    : [H, W]
        target["ch_mask"]    : [H, W]
        target["eq_mean"]    : [2]
        target["eq_std"]     : [2]
        target["ch_mean"]    : [2]
        target["ch_std"]     : [2]
        target["dmrs_mask"]  : [H, W]
        target["data_mask"]  : [H, W]

    cfg : dict
        parsed ldpc_cfg.txt

    bdir : str
        bundle directory path
    """

    RE_EQ = re.compile(r"^eqsymb_sym(\d{2})\.bin$")
    RE_CH = re.compile(r"^chestext_sym(\d{2})\.bin$")

    def __init__(
        self,
        bundle_root,
        limit=None,
        normalize=True,
        aux_normalize=True,
        aux_normalize_eq=None,
        aux_normalize_ch=None,
        eps=1e-6,
        T=1,
        keep_rect=True,
        require_aux_targets=True,
    ):
        super().__init__()
        self.bundle_root = bundle_root
        self.normalize = normalize

        # backward compatible:
        # old code only had aux_normalize=True/False
        if aux_normalize_eq is None:
            aux_normalize_eq = aux_normalize
        if aux_normalize_ch is None:
            aux_normalize_ch = aux_normalize

        self.aux_normalize_eq = bool(aux_normalize_eq)
        self.aux_normalize_ch = bool(aux_normalize_ch)

        self.eps = eps
        self.T = T
        self.keep_rect = keep_rect
        self.require_aux_targets = require_aux_targets

        dirs = sorted(
            d for d in os.listdir(bundle_root)
            if d.startswith("f") and os.path.isdir(os.path.join(bundle_root, d))
        )

        # -------------------------------------------------
        # Drop fixed PHY warm-up transient bundles
        # Empirically, each run's first 6 bundles are non-steady-state:
        #   fullgrid absmax ~= 338~340, then from bundle #7 onward ~= 263~265
        # So we remove the first 6 bundles before any train/val split.
        # -------------------------------------------------
        num_warmup_drop = 6
        n_total_before_drop = len(dirs)

        if len(dirs) > num_warmup_drop:
            dirs = dirs[num_warmup_drop:]
        else:
            dirs = []

        if limit is not None:
            dirs = dirs[:limit]

        self.bundle_dirs = [os.path.join(bundle_root, d) for d in dirs]

        print(f"[Dataset] Found {n_total_before_drop} bundles before warm-up drop")
        print(f"[Dataset] Dropped first {min(num_warmup_drop, n_total_before_drop)} warm-up bundles")
        print(f"[Dataset] Using {len(self.bundle_dirs)} bundles after warm-up drop")
        print(
            f"[Dataset] normalize={self.normalize} "
            f"aux_normalize_eq={self.aux_normalize_eq} "
            f"aux_normalize_ch={self.aux_normalize_ch}"
        )

    def __len__(self):
        return len(self.bundle_dirs)

    def __getitem__(self, idx):
        bdir = self.bundle_dirs[idx]

        # -------------------------------------------------
        # ldpc cfg / llr
        # -------------------------------------------------
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

        # -------------------------------------------------
        # fullgrid -> [1, T, 2, H, W]
        # -------------------------------------------------
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
        if tuple(x.shape) != expected_shape:
            raise RuntimeError(
                f"[ERROR] unexpected x shape in {bdir}: "
                f"got {tuple(x.shape)}, expected {expected_shape}"
            )

        # -------------------------------------------------
        # pdsch cfg / masks
        # -------------------------------------------------
        pdsch_cfg_path = os.path.join(bdir, "pdsch_cfg.txt")
        if not os.path.exists(pdsch_cfg_path):
            raise FileNotFoundError(f"[ERROR] pdsch_cfg.txt missing: {bdir}")

        pdsch_cfg = self._parse_kv_file(pdsch_cfg_path)

        dmrs_mask, data_mask, dmrs_symbols = self._build_dmrs_and_data_mask(
            pdsch_cfg=pdsch_cfg,
            meta=meta,
            bdir=bdir,
        )  # [H, W], [H, W], [list]

        dmrs_mask_t = dmrs_mask.unsqueeze(0).repeat(self.T, 1, 1)
        data_mask_t = data_mask.unsqueeze(0).repeat(self.T, 1, 1)

        # -------------------------------------------------
        # normalize input RI channels only
        # -------------------------------------------------
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
                dmrs_mask_t.unsqueeze(1),  # [T,1,H,W]
                data_mask_t.unsqueeze(1),  # [T,1,H,W]
            ],
            dim=1,
        )  # [T,4,H,W]

        expected_out_shape = (self.T, 4, H, W)
        if tuple(x_out.shape) != expected_out_shape:
            raise RuntimeError(
                f"[ERROR] unexpected output x shape in {bdir}: "
                f"got {tuple(x_out.shape)}, expected {expected_out_shape}"
            )

        # -------------------------------------------------
        # eq / channel dense targets on [2, H, W]
        # -------------------------------------------------
        y_eq_raw, eq_mask = self._load_dense_symbol_targets(
            bdir=bdir,
            H=H,
            W=W,
            symbol_file_kind="eq",
            data_mask=data_mask,
        )

        y_ch_raw, ch_mask = self._load_dense_symbol_targets(
            bdir=bdir,
            H=H,
            W=W,
            symbol_file_kind="ch",
            data_mask=data_mask,
        )

        if self.require_aux_targets:
            if eq_mask.sum().item() <= 0:
                raise RuntimeError(f"[ERROR] no eq supervision found in {bdir}")
            if ch_mask.sum().item() <= 0:
                raise RuntimeError(f"[ERROR] no channel supervision found in {bdir}")

        # -------------------------------------------------
        # normalize eq/ch independently
        # -------------------------------------------------
        if self.aux_normalize_eq:
            y_eq, eq_mean, eq_std = self._masked_channelwise_normalize(y_eq_raw, eq_mask)
        else:
            y_eq = y_eq_raw
            eq_mean = torch.zeros(2, dtype=torch.float32)
            eq_std = torch.ones(2, dtype=torch.float32)

        if self.aux_normalize_ch:
            y_ch, ch_mean, ch_std = self._masked_channelwise_normalize(y_ch_raw, ch_mask)
        else:
            y_ch = y_ch_raw
            ch_mean = torch.zeros(2, dtype=torch.float32)
            ch_std = torch.ones(2, dtype=torch.float32)

        target = {
            "y_llr": y_llr,
            "y_eq": y_eq,
            "y_ch": y_ch,
            "eq_mask": eq_mask,
            "ch_mask": ch_mask,
            "eq_mean": eq_mean,
            "eq_std": eq_std,
            "ch_mean": ch_mean,
            "ch_std": ch_std,
            "dmrs_mask": dmrs_mask,
            "data_mask": data_mask,
            "dmrs_symbols": torch.tensor(dmrs_symbols, dtype=torch.int64),
        }

        return x_out, target, cfg, bdir

    # =================================================
    # basic parsers
    # =================================================
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
                    v = int(v, 0)
                except Exception:
                    pass

                cfg[k] = v

        return cfg

    # =================================================
    # mask builder
    # =================================================
    def _build_dmrs_and_data_mask(self, pdsch_cfg, meta, bdir):
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
            raise RuntimeError(f"[ERROR] pdsch_cfg missing keys {missing} in {bdir}")

        dlDmrsSymbPos = int(pdsch_cfg["dlDmrsSymbPos"])
        dmrsConfigType = int(pdsch_cfg["dmrsConfigType"])
        n_dmrs_cdm_groups = int(pdsch_cfg["n_dmrs_cdm_groups"])
        start_symbol = int(pdsch_cfg["start_symbol"])
        number_symbols = int(pdsch_cfg["number_symbols"])
        number_rbs = int(pdsch_cfg["number_rbs"])

        BWPStart = int(pdsch_cfg.get("BWPStart", 0))
        start_rb = int(pdsch_cfg.get("start_rb", 0))

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

        local_total_rbs = W // 12
        if W % 12 != 0:
            raise RuntimeError(f"[ERROR] used_sc={W} is not divisible by 12 in {bdir}")

        local_start_rb = BWPStart + start_rb
        local_number_rbs = number_rbs

        if local_start_rb < 0 or local_number_rbs <= 0:
            raise RuntimeError(
                f"[ERROR] invalid RB range in {bdir}: "
                f"BWPStart={BWPStart}, start_rb={start_rb}, number_rbs={number_rbs}"
            )

        if local_start_rb + local_number_rbs > local_total_rbs:
            raise RuntimeError(
                f"[ERROR] active RB range exceeds used-grid in {bdir}: "
                f"local_start_rb={local_start_rb}, local_number_rbs={local_number_rbs}, "
                f"local_total_rbs={local_total_rbs}"
            )

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

        return dmrs_mask, data_mask, dmrs_symbols

    # =================================================
    # supervision loading
    # =================================================
    def _load_dense_symbol_targets(self, bdir, H, W, symbol_file_kind, data_mask):
        if symbol_file_kind == "eq":
            rx = self.RE_EQ
            prefix = "eqsymb"
        elif symbol_file_kind == "ch":
            rx = self.RE_CH
            prefix = "chestext"
        else:
            raise ValueError(f"unknown symbol_file_kind={symbol_file_kind}")

        sym_files = {}
        for fn in os.listdir(bdir):
            m = rx.match(fn)
            if m:
                sym = int(m.group(1))
                sym_files[sym] = os.path.join(bdir, fn)

        dense = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((H, W), dtype=torch.float32)

        for s in sorted(sym_files.keys()):
            path = sym_files[s]
            vals_ri = self._load_packed_c16_like_as_ri(path)  # [N, 2]
            data_pos = torch.nonzero(data_mask[s] > 0.5, as_tuple=False).squeeze(1)

            n_file = int(vals_ri.shape[0])
            n_mask = int(data_pos.numel())

            if n_file != n_mask:
                raise RuntimeError(
                    f"[ERROR] {prefix} length mismatch in {bdir}, sym={s:02d}: "
                    f"file has {n_file} active RE, but data_mask has {n_mask}"
                )

            if n_file == 0:
                continue

            sc_idx = data_pos.long()
            dense[0, s, sc_idx] = vals_ri[:, 0]
            dense[1, s, sc_idx] = vals_ri[:, 1]
            valid_mask[s, sc_idx] = 1.0

        return dense, valid_mask

    def _load_packed_c16_like_as_ri(self, path):
        raw = np.fromfile(path, dtype=np.int16)
        if raw.size % 2 != 0:
            raise RuntimeError(
                f"[ERROR] packed complex file has odd int16 count: {path}"
            )

        ri = raw.reshape(-1, 2).astype(np.float32)
        return torch.from_numpy(ri)

    def _masked_channelwise_normalize(self, y: torch.Tensor, mask: torch.Tensor):
        """
        y    : [2, H, W]
        mask : [H, W]

        normalize each RI channel using masked mean/std only.
        """
        out = y.clone()
        mean = torch.zeros(2, dtype=torch.float32)
        std = torch.ones(2, dtype=torch.float32)

        valid = mask > 0.5
        n = int(valid.sum().item())
        if n <= 0:
            return out, mean, std

        for c in range(2):
            vals = y[c][valid]
            m = vals.mean()
            s = vals.std(unbiased=False)

            if not torch.isfinite(m):
                m = torch.tensor(0.0, dtype=torch.float32)
            if not torch.isfinite(s) or s.item() < self.eps:
                s = torch.tensor(1.0, dtype=torch.float32)

            out[c][valid] = (y[c][valid] - m) / (s + self.eps)
            mean[c] = m
            std[c] = s

        return out, mean, std
