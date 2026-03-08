#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import subprocess
import numpy as np

def read_int_kv(cfg_txt: Path, key: str) -> int:
    for ln in cfg_txt.read_text().splitlines():
        p = ln.strip().split()
        if len(p) == 2 and p[0] == key:
            return int(p[1])
    raise RuntimeError(f"{key} not found in {cfg_txt}")

def float32_llr_to_int8(llr_f32: np.ndarray, scale: float) -> np.ndarray:
    x = np.rint(llr_f32 * scale)
    return np.clip(x, -127, 127).astype(np.int8)

def load_oai_llr(path: Path, G: int) -> np.ndarray:
    nbytes = path.stat().st_size
    if nbytes == G * 4:
        return np.fromfile(path, dtype=np.float32)
    if nbytes == G * 2:
        return np.fromfile(path, dtype=np.int16).astype(np.float32)
    # fallback: try float32 anyway, but warn
    arr = np.fromfile(path, dtype=np.float32)
    return arr

def unpack_txbits(txbits_bin: Path, A: int) -> np.ndarray:
    tx_bytes = np.fromfile(txbits_bin, dtype=np.uint8)
    tx_bits = np.unpackbits(tx_bytes)[:A].astype(np.uint8)
    return tx_bits

def read_decoded_bits(decoded_bits: Path, A: int) -> np.ndarray:
    dec = np.fromfile(decoded_bits, dtype=np.uint8)
    # ldpctest_spx 輸出通常是 0/1 bytes；保險起見只取 LSB
    dec = (dec.astype(np.uint8) & 1)[:A]
    return dec

def ber(txbits_bin: Path, decoded_bits: Path, A: int) -> float:
    tx_bytes = np.fromfile(txbits_bin, dtype=np.uint8)
    tx_bits = np.unpackbits(tx_bytes).astype(np.uint8)

    dec = np.fromfile(decoded_bits, dtype=np.uint8).astype(np.uint8)

    # decoded_bits is 1 bit -> 1 byte in ldpctest_spx output
    if dec.size < A:
        return 1.0

    if tx_bits.size < A:
        # txbits too short => definitely mismatch
        return 1.0

    tx_bits = tx_bits[:A]
    dec = dec[:A]

    # now guaranteed same length
    return float(np.mean(tx_bits != dec))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oai_llr_bin", type=str)
    ap.add_argument("ldpc_cfg_txt", type=str)
    ap.add_argument("txbits_bin", type=str)
    ap.add_argument("--ldpctest", type=str, required=True)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--out", type=str, default="decoded_bits_oai.bin")
    args = ap.parse_args()

    oai_llr_bin = Path(args.oai_llr_bin)
    cfg_txt     = Path(args.ldpc_cfg_txt)
    txbits_bin  = Path(args.txbits_bin)
    out_bits    = Path(args.out)
    ldpctest    = Path(args.ldpctest)

    A = read_int_kv(cfg_txt, "A")
    G = read_int_kv(cfg_txt, "G")

    llr = load_oai_llr(oai_llr_bin, G)
    llr = np.ravel(llr)
    if llr.size != G:
        print(f"[WARN] LLR length mismatch: got {llr.size}, expected G={G} (file bytes={oai_llr_bin.stat().st_size})")

    llr_i8 = float32_llr_to_int8(llr.astype(np.float32), args.scale)
    tmp_i8 = oai_llr_bin.parent / f"oai_llr_int8_s{int(args.scale)}.bin"
    llr_i8.tofile(tmp_i8)

    cmd = [str(ldpctest), str(tmp_i8), str(cfg_txt), str(out_bits)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # --- diagnostics ---
    tx_bits = unpack_txbits(txbits_bin, A)
    dec_bits = read_decoded_bits(out_bits, A)

    b = ber_bits(tx_bits, dec_bits)

    print(f"[DIAG] A={A} G={G}")
    print(f"[DIAG] oai_llr bytes={oai_llr_bin.stat().st_size} llr_len={llr.size} llr_min={float(llr.min()):.3f} llr_max={float(llr.max()):.3f} llr_std={float(llr.std()):.3f}")
    print(f"[DIAG] int8 nonzero={float(np.mean(llr_i8!=0)):.3f} abs>=10={float(np.mean(np.abs(llr_i8)>=10)):.3f} min={int(llr_i8.min())} max={int(llr_i8.max())}")
    print(f"[DIAG] txbits bytes={txbits_bin.stat().st_size} tx_bits_len={tx_bits.size}")
    print(f"[DIAG] decoded_bits bytes={out_bits.stat().st_size if out_bits.exists() else -1} dec_bits_len={dec_bits.size}")
    print(f"[OAI-DECODE] scale={args.scale} rc={p.returncode} BER={b:.6f}")
    print(p.stdout[-1200:])

    meta = {
        "scale": args.scale,
        "rc": p.returncode,
        "BER": b,
        "cmd": cmd,
        "A": A,
        "G": G,
        "paths": {
            "oai_llr_bin": str(oai_llr_bin),
            "ldpc_cfg_txt": str(cfg_txt),
            "txbits_bin": str(txbits_bin),
            "tmp_i8": str(tmp_i8),
            "out_bits": str(out_bits),
        },
        "sizes": {
            "oai_llr_bytes": oai_llr_bin.stat().st_size,
            "txbits_bytes": txbits_bin.stat().st_size,
            "out_bits_bytes": out_bits.stat().st_size if out_bits.exists() else -1,
        },
        "llr_stats": {
            "min": float(llr.min()) if llr.size else None,
            "max": float(llr.max()) if llr.size else None,
            "std": float(llr.std()) if llr.size else None,
            "len": int(llr.size),
        },
    }
    (oai_llr_bin.parent / "oai_decode_meta.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
