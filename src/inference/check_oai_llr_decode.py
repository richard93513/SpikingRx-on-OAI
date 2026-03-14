#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
from pathlib import Path
import numpy as np


def read_int_kv(cfg_txt: Path, key: str) -> int:
    for ln in cfg_txt.read_text().splitlines():
        p = ln.strip().split()
        if len(p) == 2 and p[0] == key:
            return int(p[1])
    raise RuntimeError(f"{key} not found in {cfg_txt}")


def unpack_txbits(txbits_bin: Path, A: int) -> np.ndarray:
    tx_bytes = np.fromfile(txbits_bin, dtype=np.uint8)
    tx_bits = np.unpackbits(tx_bytes, bitorder="big")[:A].astype(np.uint8)
    return tx_bits


def read_decoded_bits(decoded_bits: Path, A: int) -> np.ndarray:
    dec = np.fromfile(decoded_bits, dtype=np.uint8)
    dec = (dec.astype(np.uint8) & 1)[:A]
    return dec


def ber_bits(tx_bits: np.ndarray, dec_bits: np.ndarray) -> float:
    if tx_bits.size < dec_bits.size:
        dec_bits = dec_bits[:tx_bits.size]
    elif dec_bits.size < tx_bits.size:
        tx_bits = tx_bits[:dec_bits.size]

    if tx_bits.size == 0 or dec_bits.size == 0:
        return 1.0

    return float(np.mean(tx_bits != dec_bits))


def file_stats(path: Path, dtype) -> dict:
    if not path.exists():
        return {"exists": False}

    arr = np.fromfile(path, dtype=dtype)
    out = {
        "exists": True,
        "bytes": path.stat().st_size,
        "len": int(arr.size),
    }

    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        out.update({
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        })

    return out


def run_cmd(cmd, cwd=None):
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p


def main():
    ap = argparse.ArgumentParser(
        description="Run OAI/SpikingRx decode pipeline: LLR -> rmunmatch_spx -> ldpctest_spx -> BER"
    )
    ap.add_argument("llr_bin", type=str, help="Input LLR file (typically demapper_llr_f32.bin or inferred LLR)")
    ap.add_argument("ldpc_cfg_txt", type=str, help="ldpc_cfg.txt")
    ap.add_argument("pdsch_cfg_txt", type=str, help="pdsch_cfg.txt")
    ap.add_argument("txbits_bin", type=str, help="txbits.bin")
    ap.add_argument("--rmunmatch", type=str, required=True, help="Path to rmunmatch_spx binary")
    ap.add_argument("--ldpctest", type=str, required=True, help="Path to ldpctest_spx binary")
    ap.add_argument("--llr-scale", type=float, default=1.0, help="Passed to rmunmatch_spx --llr-scale")
    ap.add_argument("--rm-prefix", type=str, default="rm_exact_spx", help="Output prefix for rmunmatch_spx")
    ap.add_argument("--out", type=str, default="decoded_bits.bin", help="Decoded bits output path")
    ap.add_argument("--meta", type=str, default="oai_decode_meta.json", help="Metadata JSON output path")
    args = ap.parse_args()

    llr_bin     = Path(args.llr_bin).resolve()
    cfg_txt     = Path(args.ldpc_cfg_txt).resolve()
    pdsch_txt   = Path(args.pdsch_cfg_txt).resolve()
    txbits_bin  = Path(args.txbits_bin).resolve()
    rmunmatch   = Path(args.rmunmatch).resolve()
    ldpctest    = Path(args.ldpctest).resolve()

    workdir = llr_bin.parent.resolve()
    rm_prefix = workdir / args.rm_prefix
    decoded_bits_out = (workdir / args.out).resolve()
    meta_path = (workdir / args.meta).resolve()

    A = read_int_kv(cfg_txt, "A")
    G = read_int_kv(cfg_txt, "G")
    C = read_int_kv(cfg_txt, "C")

    # 預期 rmunmatch 會輸出這些檔
    rm_seg00 = workdir / f"{args.rm_prefix}_seg00_i8.bin"
    rm_seg01 = workdir / f"{args.rm_prefix}_seg01_i8.bin"

    # 清舊輸出
    for p in [rm_seg00, rm_seg01, decoded_bits_out,
              workdir / "cb_payload_seg00.bin",
              workdir / "cb_payload_seg01.bin",
              workdir / "tb_payload_bytes.bin"]:
        if p.exists():
            p.unlink()

    # ---------- Step 1: rmunmatch_spx ----------
    cmd_rm = [
        str(rmunmatch),
        str(llr_bin),
        str(cfg_txt),
        str(pdsch_txt),
        str(rm_prefix),
        "--llr-scale",
        str(args.llr_scale),
    ]
    p_rm = run_cmd(cmd_rm, cwd=workdir)

    # ---------- Step 2: ldpctest_spx ----------
    # 依照你目前成功驗證的流程，直接餵 seg00 給 ldpctest_spx
    cmd_ldpc = [
        str(ldpctest),
        str(rm_seg00),
        str(cfg_txt),
        str(decoded_bits_out),
    ]
    p_ldpc = run_cmd(cmd_ldpc, cwd=workdir)

    # ---------- Step 3: BER ----------
    tx_bits = unpack_txbits(txbits_bin, A)
    dec_bits = read_decoded_bits(decoded_bits_out, A) if decoded_bits_out.exists() else np.array([], dtype=np.uint8)
    b = ber_bits(tx_bits, dec_bits)
    bit_errors = int(np.sum(tx_bits[:dec_bits.size] != dec_bits[:tx_bits.size])) if dec_bits.size > 0 else A

    # ---------- Optional oracle cmp ----------
    ue_tb = workdir / "ue_tb.bin"
    tb_payload = workdir / "tb_payload_bytes.bin"
    rc_tb = None
    if ue_tb.exists() and tb_payload.exists():
        cmp_tb = run_cmd(["cmp", "-s", str(tb_payload), str(ue_tb)], cwd=workdir)
        rc_tb = cmp_tb.returncode

    ue_c0 = workdir / "ue_c_seg00.bin"
    ue_c1 = workdir / "ue_c_seg01.bin"
    cb0 = workdir / "cb_payload_seg00.bin"
    cb1 = workdir / "cb_payload_seg01.bin"
    rc_cb0 = None
    rc_cb1 = None
    if ue_c0.exists() and cb0.exists():
        rc_cb0 = run_cmd(["cmp", "-s", str(cb0), str(ue_c0)], cwd=workdir).returncode
    if ue_c1.exists() and cb1.exists():
        rc_cb1 = run_cmd(["cmp", "-s", str(cb1), str(ue_c1)], cwd=workdir).returncode

    # ---------- Diagnostics ----------
    llr_stats = file_stats(llr_bin, np.float32 if llr_bin.stat().st_size == G * 4 else np.int16)
    rm0_stats = file_stats(rm_seg00, np.int8)
    rm1_stats = file_stats(rm_seg01, np.int8)
    dec_stats = file_stats(decoded_bits_out, np.uint8)

    print(f"[PIPE] workdir={workdir}")
    print(f"[PIPE] A={A} G={G} C={C}")
    print(f"[PIPE] llr_bin={llr_bin.name}")
    print(f"[PIPE] rm_prefix={args.rm_prefix}")
    print(f"[PIPE] decoded_bits_out={decoded_bits_out.name}")
    print()

    print(f"[RMUNMATCH] rc={p_rm.returncode}")
    print(p_rm.stdout[-2000:] if p_rm.stdout else "")

    print(f"[LDPCTEST] rc={p_ldpc.returncode}")
    print(p_ldpc.stdout[-2000:] if p_ldpc.stdout else "")

    print(f"[DIAG] llr_stats={llr_stats}")
    print(f"[DIAG] rm_seg00_stats={rm0_stats}")
    print(f"[DIAG] rm_seg01_stats={rm1_stats}")
    print(f"[DIAG] decoded_bits_stats={dec_stats}")
    print(f"[DIAG] tx_bits_len={tx_bits.size} dec_bits_len={dec_bits.size}")
    print(f"[RESULT] BER={b:.6f} bit_errors={bit_errors}/{A}")

    if rc_cb0 is not None:
        print(f"[ORACLE] rc_cb0={rc_cb0}")
    if rc_cb1 is not None:
        print(f"[ORACLE] rc_cb1={rc_cb1}")
    if rc_tb is not None:
        print(f"[ORACLE] rc_tb={rc_tb}")

    meta = {
        "A": A,
        "G": G,
        "C": C,
        "llr_scale": args.llr_scale,
        "ber": b,
        "bit_errors": bit_errors,
        "rmunmatch_rc": p_rm.returncode,
        "ldpctest_rc": p_ldpc.returncode,
        "oracle": {
            "rc_cb0": rc_cb0,
            "rc_cb1": rc_cb1,
            "rc_tb": rc_tb,
        },
        "paths": {
            "llr_bin": str(llr_bin),
            "ldpc_cfg_txt": str(cfg_txt),
            "pdsch_cfg_txt": str(pdsch_txt),
            "txbits_bin": str(txbits_bin),
            "rm_seg00": str(rm_seg00),
            "rm_seg01": str(rm_seg01),
            "decoded_bits_out": str(decoded_bits_out),
        },
        "cmd": {
            "rmunmatch": cmd_rm,
            "ldpctest": cmd_ldpc,
        },
        "stats": {
            "llr": llr_stats,
            "rm_seg00": rm0_stats,
            "rm_seg01": rm1_stats,
            "decoded_bits": dec_stats,
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[META] wrote {meta_path}")


if __name__ == "__main__":
    main()
