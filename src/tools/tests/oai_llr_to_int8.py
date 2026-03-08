#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--clip", type=float, default=32.0)
    ap.add_argument("--G", type=int, default=14400)
    args = ap.parse_args()

    bdir = Path(args.bundle_dir)
    oai_path = bdir / "oai_llr.bin"
    out_path = bdir / "oai_llr_int8.bin"

    llr = np.fromfile(oai_path, dtype=np.float32)
    assert len(llr) == args.G, f"LLR len {len(llr)} != G={args.G}"

    llr_norm = np.clip(llr / args.clip, -1.0, 1.0)
    llr_int8 = np.round(llr_norm * 127).astype(np.int8)
    llr_int8.tofile(out_path)

    print(f"[OK] wrote {out_path} (clip={args.clip})")

if __name__ == "__main__":
    main()

