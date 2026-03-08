#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np


def read_A(cfg_txt: Path) -> int:
    A = None
    for ln in cfg_txt.read_text().splitlines():
        p = ln.strip().split()
        if len(p) == 2 and p[0] == "A":
            A = int(p[1]); break
    if A is None:
        raise RuntimeError("A not found")
    return A


def ber(txbits_bin: Path, decoded_bits: Path, A: int) -> float:
    tx_bytes = np.fromfile(txbits_bin, dtype=np.uint8)
    tx_bits = np.unpackbits(tx_bytes)[:A].astype(np.uint8)
    dec = np.fromfile(decoded_bits, dtype=np.uint8)[:A].astype(np.uint8)
    if dec.size < A:
        return 1.0
    return float((tx_bits != dec).mean())


def oai_llr_gate(ldpctest: Path, oai_llr_f32: Path, cfg_txt: Path, txbits: Path, scale: float = 16.0) -> float:
    A = read_A(cfg_txt)
    llr = np.fromfile(oai_llr_f32, dtype=np.float32)
    llr_i8 = np.round(llr * scale)
    llr_i8 = np.clip(llr_i8, -127, 127).astype(np.int8)
    tmp_i8 = oai_llr_f32.parent / "_tmp_oai_llr_i8.bin"
    llr_i8.tofile(tmp_i8)

    out_bits = oai_llr_f32.parent / "_tmp_dec_oai.bin"
    cmd = [str(ldpctest), str(tmp_i8), str(cfg_txt), str(out_bits)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        return 1.0
    return ber(txbits, out_bits, A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_root", type=str, default="spx_records/bundle")
    ap.add_argument("--ldpctest", type=str, required=True)
    ap.add_argument("--oai_gate_scale", type=float, default=16.0)
    ap.add_argument("--oai_gate_th", type=float, default=0.45)
    args = ap.parse_args()

    bundle_root = Path(args.bundle_root)
    ldpctest = Path(args.ldpctest)

    results = []
    for bdir in sorted(bundle_root.glob("f*_s*_idx*")):
        fullgrid = bdir / "fullgrid.bin"
        oai_llr  = bdir / "oai_llr.bin"
        cfg_txt  = bdir / "ldpc_cfg.txt"
        txbits   = bdir / "txbits.bin"

        if not (fullgrid.exists() and oai_llr.exists() and cfg_txt.exists() and txbits.exists()):
            continue

        ber_oai = oai_llr_gate(ldpctest, oai_llr, cfg_txt, txbits, scale=args.oai_gate_scale)
        status = "OK" if ber_oai <= args.oai_gate_th else "BUNDLE_MISMATCH"

        meta = {
            "bundle": bdir.name,
            "ber_oai": ber_oai,
            "status": status,
        }
        (bdir / "gate_meta.json").write_text(json.dumps(meta, indent=2))
        results.append(meta)
        print(f"[{status}] {bdir.name}  BER_oai={ber_oai:.6f}")

    (bundle_root / "batch_gate_summary.json").write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote summary -> {bundle_root/'batch_gate_summary.json'}")


if __name__ == "__main__":
    main()

