#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpikingRx-on-OAI bundler (RAW -> BUNDLE)

This version is aligned to the NEW dump naming:
- UE:
  - f%04d_s%02d_fullgrid_idx%06d_rnti%04x_harq%02d.bin
  - f%04d_s%02d_llr_idx%06d_rnti%04x_harq%02d_f32.bin
  - f%04d_s%02d_ldpc_idx%06d_rnti%04x_dlsch%02d_harq%02d_round%02d_rv%d.json
  - f%04d_s%02d_ue_tb_idx%06d_rnti%04x_dlsch%02d_harq%02d_round%02d_rv%d.bin
  - f%04d_s%02d_ue_c_idx%06d_rnti%04x_dlsch%02d_harq%02d_round%02d_rv%d_seg%02d.bin
  - f%04d_s%02d_ue_llr_preunscram_rnti%05d_harq%02d_cw%d.bin
  - f%04d_s%02d_ue_llr_postunscram_rnti%05d_harq%02d_cw%d.bin
  - f%04d_s%02d_pdsch_cfg_rnti%05d_harq%02d_cw%d.txt
  - f%04d_s%02d_ldpc_rm_exact_idx%06d_tb%02d_rv%d_seg%02d_outlen%05d_i8.bin
  - f%04d_s%02d_ldpc_ue_dunmatch_idx%06d_tb%02d_rv%d_seg%02d_len%05d_i16.bin
  - f%04d_s%02d_ldpc_ue_z_idx%06d_tb%02d_rv%d_seg%02d_len%05d_i16.bin
  - f%04d_s%02d_ldpc_ue_l_idx%06d_tb%02d_rv%d_seg%02d_outlen%05d_i8.bin
- gNB:
  - f%04d_s%02d_txbits_idx%06d_rnti%04x_pdu%03d_rv%d_tbcrc%08x.bin
  - f%04d_s%02d_ldpc_idx%06d_rnti%04x_pdu%03d_rv%d_tbcrc%08x.json

Notes:
- rnti is treated as HEX by default for 4-hex-digit strings; still accepts 5-digit decimal.
- ONE UE record is defined by (frame,slot,idx,rnti,harq).
- It reads UE LDPC json to obtain (dlsch_id, round, rv_index).
- rm_exact / ue_dunmatch / ue_z / ue_l use tb=0 by default for current single-TB case.
- unscrambling dumps are matched by (frame,slot,rnti,harq,cw), because these filenames do not carry idx.
"""

import re
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any


REPO_ROOT = Path.home() / "SpikingRx-on-OAI"
RAW_DIR = REPO_ROOT / "spx_records/raw"
BUNDLE_DIR = REPO_ROOT / "spx_records/bundle"


# ------------------------------
# Helpers
# ------------------------------

def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path) -> dict:
    return json.loads(p.read_text())

def parse_rnti(s: str) -> int:
    """
    Accept:
      - 4-hex-digit: "1234" or "0abc" or "ABCD"
      - 5-decimal-digit: "04660"
    Strategy:
      - if contains any hex letter -> base16
      - else if length==4 -> base16
      - else -> base10
    """
    ss = s.strip()
    has_hex_letter = any(c in "abcdefABCDEF" for c in ss)
    if has_hex_letter or len(ss) == 4:
        return int(ss, 16)
    return int(ss, 10)

def safe_int(d: Dict[str, Any], k: str, default: int = -1) -> int:
    try:
        return int(d.get(k, default))
    except Exception:
        return default

def load_kv_txt(p: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split(None, 1)
        if len(parts) != 2:
            continue
        k, v = parts
        try:
            out[k] = int(v, 0)
        except Exception:
            out[k] = v
    return out

# ------------------------------
# Regex
# ------------------------------

RE_UE_FULLGRID = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_fullgrid_idx(?P<idx>\d{6})_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_harq(?P<harq>\d{2})\.bin$"
)
RE_UE_LLR = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_llr_idx(?P<idx>\d{6})_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_harq(?P<harq>\d{2})(?:_f32)?\.bin$"
)
RE_UE_LDPC = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ldpc_idx(?P<idx>\d{6})_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_dlsch(?P<dlsch>\d{2})_harq(?P<harq>\d{2})_round(?P<round>\d{2})_rv(?P<rv>\d)\.json$"
)
RE_UE_TB = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ue_tb_idx(?P<idx>\d{6})_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_dlsch(?P<dlsch>\d{2})_harq(?P<harq>\d{2})_round(?P<round>\d{2})_rv(?P<rv>\d)\.bin$"
)
RE_UE_C = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ue_c_idx(?P<idx>\d{6})_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_dlsch(?P<dlsch>\d{2})_harq(?P<harq>\d{2})_round(?P<round>\d{2})_rv(?P<rv>\d)_seg(?P<seg>\d{2})\.bin$"
)

RE_UE_RM_EXACT = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ldpc_rm_exact_idx(?P<idx>\d{6})_tb(?P<tb>\d{2})_rv(?P<rv>\d)_seg(?P<seg>\d{2})_outlen(?P<outlen>\d{5})_i8\.bin$"
)
RE_UE_DUNMATCH = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ldpc_ue_dunmatch_idx(?P<idx>\d{6})_tb(?P<tb>\d{2})_rv(?P<rv>\d)_seg(?P<seg>\d{2})_len(?P<len>\d{5})_i16\.bin$"
)
RE_UE_Z = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ldpc_ue_z_idx(?P<idx>\d{6})_tb(?P<tb>\d{2})_rv(?P<rv>\d)_seg(?P<seg>\d{2})_len(?P<len>\d{5})_i16\.bin$"
)
RE_UE_L = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ldpc_ue_l_idx(?P<idx>\d{6})_tb(?P<tb>\d{2})_rv(?P<rv>\d)_seg(?P<seg>\d{2})_outlen(?P<outlen>\d{5})_i8\.bin$"
)

RE_UE_DECINLLR = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ue_decinllr_idx(?P<idx>\d{6})_dlsch(?P<dlsch>\d{2})_harq(?P<harq>\d{2})_round(?P<round>\d{2})_rv(?P<rv>\d)\.bin$"
)

RE_UE_PREUNSCRAM = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ue_llr_preunscram_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_harq(?P<harq>\d{2})_cw(?P<cw>\d)\.bin$"
)

RE_UE_POSTUNSCRAM = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ue_llr_postunscram_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_harq(?P<harq>\d{2})_cw(?P<cw>\d)\.bin$"
)

RE_UE_PDSCH_CFG = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_pdsch_cfg_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_harq(?P<harq>\d{2})_cw(?P<cw>\d)\.txt$"
)

RE_GNB_TXBITS = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_txbits_idx(?P<idx>\d{6})_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_pdu(?P<pdu>\d{3})_rv(?P<rv>\d)_tbcrc(?P<tbcrc>[0-9a-fA-F]{8})\.bin$"
)
RE_GNB_LDPC = re.compile(
    r"^f(?P<frame>\d{4})_s(?P<slot>\d{2})_ldpc_idx(?P<idx>\d{6})_rnti(?P<rnti>[0-9a-fA-F]{4}|\d{5})_pdu(?P<pdu>\d{3})_rv(?P<rv>\d)_tbcrc(?P<tbcrc>[0-9a-fA-F]{8})\.json$"
)


# ------------------------------
# Keys / Packs
# ------------------------------

@dataclass(frozen=True)
class UEKeyR:
    frame: int
    slot: int
    idx: int
    rnti: int
    harq: int

@dataclass(frozen=True)
class UEFullKey:
    frame: int
    slot: int
    idx: int
    rnti: int
    dlsch: int
    harq: int
    round: int
    rv: int

@dataclass(frozen=True)
class GNBKey:
    frame: int
    slot: int
    idx: int
    rnti: int
    pdu: int
    rv: int
    tbcrc: str

@dataclass
class GNBPack:
    key: GNBKey
    txbits: Path
    gnb_ldpc: Optional[Path]


# ------------------------------
# LDPC cfg conversion
# ------------------------------

def ldpc_json_to_cfg_txt(ue_ldpc_json: Path, out_cfg_txt: Path) -> Dict[str, int]:
    j = load_json(ue_ldpc_json)

    need = ["BG", "Zc", "A", "C", "K", "F", "G", "Qm", "tbslbrm", "mcs", "rv_index", "nb_layers"]
    kv: Dict[str, int] = {}
    for k in need:
        if k not in j:
            raise KeyError(f"{ue_ldpc_json.name} missing key '{k}'")
        kv[k] = int(j[k])

    if "R_list" in j and isinstance(j["R_list"], list) and len(j["R_list"]) > 0:
        kv["R"] = int(j["R_list"][0])
    else:
        raise KeyError(f"{ue_ldpc_json.name} missing R_list for R")

    order = ["BG", "Zc", "A", "C", "K", "F", "G", "Qm", "tbslbrm", "mcs", "rv_index", "nb_layers", "R"]
    out_cfg_txt.write_text("".join([f"{k} {kv[k]}\n" for k in order]))
    return kv


def pick_best_gnb_candidate(cands: List[GNBPack]) -> Optional[GNBPack]:
    if not cands:
        return None

    def score(c: GNBPack) -> Tuple[int, int, float]:
        tbcrc_ok = 1 if c.key.tbcrc.lower() == "00000000" else 0
        pdu_ok = 1 if c.key.pdu == 0 else 0
        mtime = c.txbits.stat().st_mtime
        return (tbcrc_ok, pdu_ok, mtime)

    return sorted(cands, key=score, reverse=True)[0]


# ------------------------------
# Scan RAW
# ------------------------------

def scan_raw():
    ue_r_parts: Dict[UEKeyR, Dict[str, Path]] = {}
    ue_ldpc_map: Dict[UEFullKey, Path] = {}
    ue_decin_map: Dict[Tuple[int, int, int, int, int, int, int], Path] = {}
    ue_tb_map: Dict[UEFullKey, Path] = {}
    ue_c_map: Dict[UEFullKey, Dict[int, Path]] = {}
    
    ue_preuns_map: Dict[Tuple[int, int, int, int, int], Path] = {}
    ue_postuns_map: Dict[Tuple[int, int, int, int, int], Path] = {}
    ue_pdsch_cfg_map: Dict[Tuple[int, int, int, int, int], Path] = {}

    ue_rm_exact_map: Dict[Tuple[int, int, int, int, int], Dict[int, Path]] = {}
    ue_dunmatch_map: Dict[Tuple[int, int, int, int, int], Dict[int, Path]] = {}
    ue_z_map: Dict[Tuple[int, int, int, int, int], Dict[int, Path]] = {}
    ue_l_map: Dict[Tuple[int, int, int, int, int], Dict[int, Path]] = {}

    gnb_groups: Dict[Tuple[int, int, int, int, int], List[GNBPack]] = {}
    gnb_ldpc_map: Dict[Tuple[int, int, int, int, int, int, str], Path] = {}

    if not RAW_DIR.exists():
        raise SystemExit(f"[ERR] RAW_DIR not found: {RAW_DIR}")

    for p in RAW_DIR.glob("*"):
        name = p.name

        m = RE_UE_FULLGRID.match(name)
        if m:
            k = UEKeyR(
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                parse_rnti(m.group("rnti")),
                int(m.group("harq")),
            )
            ue_r_parts.setdefault(k, {})["fullgrid"] = p
            continue

        m = RE_UE_LLR.match(name)
        if m:
            k = UEKeyR(
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                parse_rnti(m.group("rnti")),
                int(m.group("harq")),
            )
            ue_r_parts.setdefault(k, {})["llr"] = p
            continue

        m = RE_UE_LDPC.match(name)
        if m:
            k = UEFullKey(
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                parse_rnti(m.group("rnti")),
                int(m.group("dlsch")),
                int(m.group("harq")),
                int(m.group("round")),
                int(m.group("rv")),
            )
            ue_ldpc_map[k] = p
            continue

        m = RE_UE_DECINLLR.match(name)
        if m:
            k = (
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                int(m.group("dlsch")),
                int(m.group("harq")),
                int(m.group("round")),
                int(m.group("rv")),
            )
            ue_decin_map[k] = p
            continue

        m = RE_UE_TB.match(name)
        if m:
            k = UEFullKey(
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                parse_rnti(m.group("rnti")),
                int(m.group("dlsch")),
                int(m.group("harq")),
                int(m.group("round")),
                int(m.group("rv")),
            )
            ue_tb_map[k] = p
            continue

        m = RE_UE_C.match(name)
        if m:
            k = UEFullKey(
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                parse_rnti(m.group("rnti")),
                int(m.group("dlsch")),
                int(m.group("harq")),
                int(m.group("round")),
                int(m.group("rv")),
            )
            seg = int(m.group("seg"))
            ue_c_map.setdefault(k, {})[seg] = p
            continue

        m = RE_UE_PREUNSCRAM.match(name)
        if m:
            k = (
                int(m.group("frame")),
                int(m.group("slot")),
                parse_rnti(m.group("rnti")),
                int(m.group("harq")),
                int(m.group("cw")),
            )
            ue_preuns_map[k] = p
            continue

        m = RE_UE_POSTUNSCRAM.match(name)
        if m:
            k = (
                int(m.group("frame")),
                int(m.group("slot")),
                parse_rnti(m.group("rnti")),
                int(m.group("harq")),
                int(m.group("cw")),
            )
            ue_postuns_map[k] = p
            continue

        m = RE_UE_PDSCH_CFG.match(name)
        if m:
            k = (
                int(m.group("frame")),
                int(m.group("slot")),
                parse_rnti(m.group("rnti")),
                int(m.group("harq")),
                int(m.group("cw")),
            )
            ue_pdsch_cfg_map[k] = p
            continue

        m = RE_UE_RM_EXACT.match(name)
        if m:
            key = (
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                int(m.group("tb")),
                int(m.group("rv")),
            )
            seg = int(m.group("seg"))
            ue_rm_exact_map.setdefault(key, {})[seg] = p
            continue

        m = RE_UE_DUNMATCH.match(name)
        if m:
            key = (
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                int(m.group("tb")),
                int(m.group("rv")),
            )
            seg = int(m.group("seg"))
            ue_dunmatch_map.setdefault(key, {})[seg] = p
            continue

        m = RE_UE_Z.match(name)
        if m:
            key = (
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                int(m.group("tb")),
                int(m.group("rv")),
            )
            seg = int(m.group("seg"))
            ue_z_map.setdefault(key, {})[seg] = p
            continue

        m = RE_UE_L.match(name)
        if m:
            key = (
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                int(m.group("tb")),
                int(m.group("rv")),
            )
            seg = int(m.group("seg"))
            ue_l_map.setdefault(key, {})[seg] = p
            continue

        m = RE_GNB_TXBITS.match(name)
        if m:
            gk = GNBKey(
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                parse_rnti(m.group("rnti")),
                int(m.group("pdu")),
                int(m.group("rv")),
                m.group("tbcrc").lower(),
            )
            pack = GNBPack(key=gk, txbits=p, gnb_ldpc=None)
            group_k = (gk.frame, gk.slot, gk.idx, gk.rnti, gk.rv)
            gnb_groups.setdefault(group_k, []).append(pack)
            continue

        m = RE_GNB_LDPC.match(name)
        if m:
            tup = (
                int(m.group("frame")),
                int(m.group("slot")),
                int(m.group("idx")),
                parse_rnti(m.group("rnti")),
                int(m.group("pdu")),
                int(m.group("rv")),
                m.group("tbcrc").lower(),
            )
            gnb_ldpc_map[tup] = p
            continue

    for packs in gnb_groups.values():
        for pack in packs:
            t = (
                pack.key.frame, pack.key.slot, pack.key.idx, pack.key.rnti,
                pack.key.pdu, pack.key.rv, pack.key.tbcrc
            )
            if t in gnb_ldpc_map:
                pack.gnb_ldpc = gnb_ldpc_map[t]

    return (
        ue_r_parts,
        ue_ldpc_map,
        ue_decin_map,
        ue_tb_map,
        ue_c_map,
        ue_preuns_map,
        ue_postuns_map,
        ue_pdsch_cfg_map,
        ue_rm_exact_map,
        ue_dunmatch_map,
        ue_z_map,
        ue_l_map,
        gnb_groups,
    )


# ------------------------------
# Build bundles
# ------------------------------

def build_bundles():
    mkdir_p(BUNDLE_DIR)

    (
        ue_r_parts,
        ue_ldpc_map,
        ue_decin_map,
        ue_tb_map,
        ue_c_map,
        ue_preuns_map,
        ue_postuns_map,
        ue_pdsch_cfg_map,
        ue_rm_exact_map,
        ue_dunmatch_map,
        ue_z_map,
        ue_l_map,
        gnb_groups,
    ) = scan_raw()

    total = 0
    built = 0
    skipped_missing_ue_core = 0
    skipped_missing_ue_ldpc = 0
    skipped_missing_ue_tb = 0
    skipped_missing_gate = 0
    skipped_missing_gnb_pair = 0
    skipped_bad_ue_ldpc_json = 0

    items = sorted(
        ue_r_parts.items(),
        key=lambda x: (x[0].frame, x[0].slot, x[0].idx, x[0].rnti, x[0].harq)
    )

    for uekr, parts in items:
        total += 1

        if "fullgrid" not in parts or "llr" not in parts:
            skipped_missing_ue_core += 1
            continue

        cand_ldpc: List[Tuple[UEFullKey, Path]] = [
            (k, p) for (k, p) in ue_ldpc_map.items()
            if (k.frame, k.slot, k.idx, k.rnti, k.harq) ==
               (uekr.frame, uekr.slot, uekr.idx, uekr.rnti, uekr.harq)
        ]
        if not cand_ldpc:
            skipped_missing_ue_ldpc += 1
            continue

        def ldpc_score(kp: Tuple[UEFullKey, Path]) -> Tuple[int, float]:
            k, p = kp
            round0 = 1 if k.round == 0 else 0
            mtime = p.stat().st_mtime
            return (round0, mtime)

        ue_fullk, ue_ldpc = sorted(cand_ldpc, key=ldpc_score, reverse=True)[0]

        try:
            ue_j = load_json(ue_ldpc)
            ue_rv_json = safe_int(ue_j, "rv_index", ue_fullk.rv)
            ue_dlsch_json = safe_int(ue_j, "dlsch_id", ue_fullk.dlsch)
            ue_round_json = safe_int(ue_j, "round", ue_fullk.round)
            if ue_rv_json < 0 or ue_dlsch_json < 0 or ue_round_json < 0:
                skipped_bad_ue_ldpc_json += 1
                continue
        except Exception:
            skipped_bad_ue_ldpc_json += 1
            continue

        ue_rv = ue_rv_json
        ue_dlsch = ue_dlsch_json
        ue_round = ue_round_json

        ue_fullk2 = UEFullKey(
            uekr.frame, uekr.slot, uekr.idx, uekr.rnti,
            ue_dlsch, uekr.harq, ue_round, ue_rv
        )

        ue_tb = ue_tb_map.get(ue_fullk2)
        if ue_tb is None:
            skipped_missing_ue_tb += 1
            continue

        ue_c_segs = ue_c_map.get(ue_fullk2, {})

        cw_id = ue_dlsch  # current single-CW path: dlsch_id == cw id

        uns_key = (
            uekr.frame,
            uekr.slot,
            uekr.rnti,
            uekr.harq,
            cw_id,
        )

        ue_preuns = ue_preuns_map.get(uns_key)
        ue_postuns = ue_postuns_map.get(uns_key)
        ue_pdsch_cfg = ue_pdsch_cfg_map.get(uns_key)

        gate_rm_exact = None
        gate_decinllr = None
        gate_rm_exact_list: List[str] = []

        dbg_dunmatch = None
        dbg_z = None
        dbg_l = None
        dbg_dunmatch_list: List[str] = []
        dbg_z_list: List[str] = []
        dbg_l_list: List[str] = []

        rm_key = (uekr.frame, uekr.slot, uekr.idx, 0, ue_rv)

        rm_segs = ue_rm_exact_map.get(rm_key, {})
        if rm_segs:
            gate_rm_exact = rm_segs
        else:
            deci_k = (uekr.frame, uekr.slot, uekr.idx, ue_dlsch, uekr.harq, ue_round, ue_rv)
            gate_decinllr = ue_decin_map.get(deci_k)

        dbg_dunmatch = ue_dunmatch_map.get(rm_key, {})
        dbg_z = ue_z_map.get(rm_key, {})
        dbg_l = ue_l_map.get(rm_key, {})

        if gate_rm_exact is None and gate_decinllr is None:
            skipped_missing_gate += 1
            continue

        group_k = (uekr.frame, uekr.slot, uekr.idx, uekr.rnti, ue_rv)
        best = pick_best_gnb_candidate(gnb_groups.get(group_k, []))
        if best is None:
            skipped_missing_gnb_pair += 1
            continue

        out_dir = BUNDLE_DIR / (
            f"f{uekr.frame:04d}_s{uekr.slot:02d}"
            f"_idx{uekr.idx:06d}"
            f"_rnti{uekr.rnti:04x}"
            f"_harq{uekr.harq:02d}"
            f"_rv{ue_rv}"
        )
        mkdir_p(out_dir)

        shutil.copy2(parts["fullgrid"], out_dir / "fullgrid.bin")
        shutil.copy2(parts["llr"], out_dir / "demapper_llr_f32.bin")
        shutil.copy2(best.txbits, out_dir / "txbits.bin")
        shutil.copy2(ue_ldpc, out_dir / "ue_ldpc.json")
        shutil.copy2(ue_tb, out_dir / "ue_tb.bin")

        if ue_preuns is not None:
            shutil.copy2(ue_preuns, out_dir / "ue_llr_preunscram_i16.bin")

        if ue_postuns is not None:
            shutil.copy2(ue_postuns, out_dir / "ue_llr_postunscram_i16.bin")

        pdsch_cfg_kv = None
        if ue_pdsch_cfg is not None:
            shutil.copy2(ue_pdsch_cfg, out_dir / "pdsch_cfg.txt")
            pdsch_cfg_kv = load_kv_txt(ue_pdsch_cfg)

        if gate_rm_exact is not None:
            for seg in sorted(gate_rm_exact.keys()):
                src = gate_rm_exact[seg]
                dst = out_dir / f"rm_exact_seg{seg:02d}_i8.bin"
                shutil.copy2(src, dst)
                gate_rm_exact_list.append(dst.name)
        else:
            shutil.copy2(gate_decinllr, out_dir / "decoder_llr_int16.bin")

        ue_c_list: List[str] = []
        for seg in sorted(ue_c_segs.keys()):
            src = ue_c_segs[seg]
            dst = out_dir / f"ue_c_seg{seg:02d}.bin"
            shutil.copy2(src, dst)
            ue_c_list.append(dst.name)

        for seg in sorted(dbg_dunmatch.keys()):
            src = dbg_dunmatch[seg]
            dst = out_dir / f"ue_dunmatch_seg{seg:02d}_i16.bin"
            shutil.copy2(src, dst)
            dbg_dunmatch_list.append(dst.name)

        for seg in sorted(dbg_z.keys()):
            src = dbg_z[seg]
            dst = out_dir / f"ue_z_seg{seg:02d}_i16.bin"
            shutil.copy2(src, dst)
            dbg_z_list.append(dst.name)

        for seg in sorted(dbg_l.keys()):
            src = dbg_l[seg]
            dst = out_dir / f"ue_l_seg{seg:02d}_i8.bin"
            shutil.copy2(src, dst)
            dbg_l_list.append(dst.name)

        cfg_kv = ldpc_json_to_cfg_txt(ue_ldpc, out_dir / "ldpc_cfg.txt")

        meta = {
            "frame": uekr.frame,
            "slot": uekr.slot,
            "idx": uekr.idx,
            "ue": {
                "rnti_hex": f"{uekr.rnti:04x}",
                "harq": uekr.harq,
                "rv": ue_rv,
                "dlsch": ue_dlsch,
                "round": ue_round,
                "fullgrid": parts["fullgrid"].name,
                "demapper_llr_f32": parts["llr"].name,
                "ue_ldpc": "ue_ldpc.json",
                "ue_tb": "ue_tb.bin",
                "ue_c_list": ue_c_list,
                "unscrambling": {
                    "cw": cw_id,
                    "preunscram_i16": ("ue_llr_preunscram_i16.bin" if ue_preuns is not None else None),
                    "postunscram_i16": ("ue_llr_postunscram_i16.bin" if ue_postuns is not None else None),
                    "pdsch_cfg": ("pdsch_cfg.txt" if ue_pdsch_cfg is not None else None),
                },
                "gate": {
                    "rm_exact_i8_list": gate_rm_exact_list if gate_rm_exact_list else None,
                    "legacy_decinllr_int16": (gate_decinllr.name if gate_decinllr else None),
                },
                "debug_oracle": {
                    "ue_dunmatch_i16_list": dbg_dunmatch_list if dbg_dunmatch_list else None,
                    "ue_z_i16_list": dbg_z_list if dbg_z_list else None,
                    "ue_l_i8_list": dbg_l_list if dbg_l_list else None,
                },
            },
            "gnb": {
                "rnti_hex": f"{best.key.rnti:04x}",
                "pdu": best.key.pdu,
                "rv": best.key.rv,
                "tbcrc": best.key.tbcrc,
                "txbits": best.txbits.name,
                "gnb_ldpc": best.gnb_ldpc.name if best.gnb_ldpc else None,
            },
            "ldpc_cfg_kv": cfg_kv,
            "pdsch_cfg_kv": pdsch_cfg_kv,
            "ue_ldpc_extra": {
                "llrLen": safe_int(ue_j, "llrLen", -1),
                "E_list": ue_j.get("E_list", None),
                "R_list": ue_j.get("R_list", None),
            },
        }

        (out_dir / "meta_bundle.json").write_text(json.dumps(meta, indent=2))
        built += 1

    summary = {
        "raw_dir": str(RAW_DIR),
        "bundle_dir": str(BUNDLE_DIR),
        "total_ue_keys": total,
        "built": built,
        "skipped_missing_ue_core": skipped_missing_ue_core,
        "skipped_missing_ue_ldpc": skipped_missing_ue_ldpc,
        "skipped_missing_ue_tb": skipped_missing_ue_tb,
        "skipped_missing_gate": skipped_missing_gate,
        "skipped_missing_gnb_pair": skipped_missing_gnb_pair,
        "skipped_bad_ue_ldpc_json": skipped_bad_ue_ldpc_json,
    }
    (BUNDLE_DIR / "batch_gate_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    s = build_bundles()
    print("[BUNDLE] raw_dir   =", s["raw_dir"])
    print("[BUNDLE] bundle_dir=", s["bundle_dir"])
    print("[BUNDLE] total UE keys =", s["total_ue_keys"])
    print("[BUNDLE] built =", s["built"])
    print("[BUNDLE] skipped_missing_ue_core =", s["skipped_missing_ue_core"])
    print("[BUNDLE] skipped_missing_ue_ldpc  =", s["skipped_missing_ue_ldpc"])
    print("[BUNDLE] skipped_missing_ue_tb    =", s["skipped_missing_ue_tb"])
    print("[BUNDLE] skipped_missing_gate     =", s["skipped_missing_gate"])
    print("[BUNDLE] skipped_missing_gnb_pair =", s["skipped_missing_gnb_pair"])
    print("[BUNDLE] skipped_bad_ue_ldpc_json =", s["skipped_bad_ue_ldpc_json"])
    print("[BUNDLE] wrote:", str(BUNDLE_DIR / "batch_gate_summary.json"))


if __name__ == "__main__":
    main()
