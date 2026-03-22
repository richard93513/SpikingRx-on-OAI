#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from data.dataset_oai_bundle import OAI_Bundle_Dataset
from models.spikingrx_model import SpikingRxModel


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def safe_tail(s: str, n: int = 4000) -> str:
    if not s:
        return ""
    return s[-n:]


def move_target_to_device(target: dict, device: torch.device):
    out = {}
    for k, v in target.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def load_model_from_ckpt(
    ckpt_path: Path,
    sample_cfg: Dict,
    device: torch.device,
) -> SpikingRxModel:
    ckpt = torch.load(ckpt_path, map_location=device)

    # 這裡跟你目前訓練成功的版本對齊：
    # - in_ch=4
    # - T=1
    # - llr_temperature=2.0
    # - out_bits = G
    model = SpikingRxModel(
        in_ch=4,
        out_bits=int(sample_cfg["G"]),
        T=1,
        device_conv=device,
        device_fc=device,
        llr_temperature=2.0,
    )

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_one_bundle_llr(
    model: SpikingRxModel,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    x: [T,4,H,W]
    return: np.float32 [G]
    """
    if x.dim() != 4:
        raise RuntimeError(f"unexpected x.dim={x.dim()}, expected 4")

    xb = x.unsqueeze(0).to(device)  # [1,T,4,H,W]
    pred, aux = model(xb)
    llr = pred["llr"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return llr


def summarize_llr(arr: np.ndarray) -> Dict:
    out = {
        "len": int(arr.size),
        "dtype": str(arr.dtype),
    }
    if arr.size > 0:
        out.update({
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        })
    return out


def maybe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Run bundle-wise SpikingRx inference: fullgrid -> model -> LLR -> check_oai_llr_decode.py"
    )
    ap.add_argument(
        "--bundle_root",
        type=str,
        default=str(Path.home() / "SpikingRx-on-OAI" / "spx_records" / "bundle"),
        help="Bundle root directory",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(Path.home() / "SpikingRx-on-OAI" / "checkpoints" / "spikingrx_oai_serial_best.pt"),
        help="Checkpoint path",
    )
    ap.add_argument(
        "--check_script",
        type=str,
        default=str(Path.home() / "SpikingRx-on-OAI" / "src" / "inference" / "check_oai_llr_decode.py"),
        help="Path to check_oai_llr_decode.py",
    )
    ap.add_argument(
        "--rmunmatch",
        type=str,
        required=True,
        help="Path to rmunmatch_spx",
    )
    ap.add_argument(
        "--ldpctest",
        type=str,
        required=True,
        help="Path to ldpctest_spx",
    )
    ap.add_argument(
        "--llr_scale",
        type=float,
        default=1.0,
        help="Passed through to check_oai_llr_decode.py --llr-scale",
    )
    ap.add_argument(
        "--pred_llr_name",
        type=str,
        default="spikingrx_llr_f32.bin",
        help="Output predicted LLR filename inside each bundle",
    )
    ap.add_argument(
        "--decoded_bits_name",
        type=str,
        default="decoded_bits_spikingrx.bin",
        help="Decoded bits filename inside each bundle",
    )
    ap.add_argument(
        "--decode_meta_name",
        type=str,
        default="spikingrx_decode_meta.json",
        help="Decode metadata filename inside each bundle",
    )
    ap.add_argument(
        "--infer_meta_name",
        type=str,
        default="spikingrx_infer_meta.json",
        help="Inference metadata filename inside each bundle",
    )
    ap.add_argument(
        "--summary_name",
        type=str,
        default="spikingrx_batch_summary.json",
        help="Batch summary JSON under bundle_root",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N bundles",
    )
    ap.add_argument(
        "--bundle_contains",
        type=str,
        default="",
        help="Only process bundle names containing this substring",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip bundles if decode meta already exists",
    )
    ap.add_argument(
        "--also_run_oai_baseline",
        action="store_true",
        help="Also call check_oai_llr_decode.py on demapper_llr_f32.bin for comparison",
    )
    ap.add_argument(
        "--oai_decoded_bits_name",
        type=str,
        default="decoded_bits_oai.bin",
        help="Decoded bits filename for OAI baseline",
    )
    ap.add_argument(
        "--oai_decode_meta_name",
        type=str,
        default="oai_decode_meta_batch.json",
        help="Metadata filename for OAI baseline",
    )
    args = ap.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    check_script = Path(args.check_script).resolve()
    rmunmatch = Path(args.rmunmatch).resolve()
    ldpctest = Path(args.ldpctest).resolve()

    if not bundle_root.exists():
        raise FileNotFoundError(f"bundle_root not found: {bundle_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not check_script.exists():
        raise FileNotFoundError(f"check_script not found: {check_script}")
    if not rmunmatch.exists():
        raise FileNotFoundError(f"rmunmatch not found: {rmunmatch}")
    if not ldpctest.exists():
        raise FileNotFoundError(f"ldpctest not found: {ldpctest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] CPU")

    # 跟目前訓練成功版一致：aux target 都 normalize
    ds = OAI_Bundle_Dataset(
        bundle_root=str(bundle_root),
        normalize=True,
        aux_normalize_eq=True,
        aux_normalize_ch=True,
        T=1,
        require_aux_targets=True,
    )

    if len(ds) == 0:
        raise RuntimeError(f"no bundles found in {bundle_root}")

    # 先拿 sample cfg 建 model
    x0, t0, cfg0, bdir0 = ds[0]
    model = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        sample_cfg=cfg0,
        device=device,
    )

    print(f"[MODEL] ckpt={ckpt_path}")
    print(f"[MODEL] sample_bundle={bdir0}")
    print(f"[MODEL] out_bits(G)={int(cfg0['G'])}")

    results = []
    processed = 0
    skipped = 0
    failed = 0

    for idx in range(len(ds)):
        x, target, cfg, bdir_str = ds[idx]
        bdir = Path(bdir_str)

        if args.bundle_contains and args.bundle_contains not in bdir.name:
            continue

        if args.limit is not None and processed >= args.limit:
            break

        pred_llr_path = bdir / args.pred_llr_name
        decoded_bits_path = bdir / args.decoded_bits_name
        decode_meta_path = bdir / args.decode_meta_name
        infer_meta_path = bdir / args.infer_meta_name

        if args.skip_existing and decode_meta_path.exists():
            print(f"[SKIP] {bdir.name}  existing={decode_meta_path.name}")
            skipped += 1
            continue

        ldpc_cfg = bdir / "ldpc_cfg.txt"
        pdsch_cfg = bdir / "pdsch_cfg.txt"
        txbits = bdir / "txbits.bin"
        demapper_llr = bdir / "demapper_llr_f32.bin"

        required = [ldpc_cfg, pdsch_cfg, txbits, demapper_llr]
        missing = [str(p.name) for p in required if not p.exists()]
        if missing:
            print(f"[FAIL] {bdir.name} missing={missing}")
            failed += 1
            results.append({
                "bundle": bdir.name,
                "status": "MISSING_FILES",
                "missing": missing,
            })
            continue

        try:
            llr_pred = infer_one_bundle_llr(
                model=model,
                x=x,
                device=device,
            )
            llr_pred.tofile(pred_llr_path)

            infer_meta = {
                "bundle": bdir.name,
                "ckpt": str(ckpt_path),
                "pred_llr_path": str(pred_llr_path),
                "pred_llr_stats": summarize_llr(llr_pred),
                "target_llr_stats": summarize_llr(target["y_llr"].detach().cpu().numpy().astype(np.float32)),
            }
            infer_meta_path.write_text(json.dumps(infer_meta, indent=2, ensure_ascii=False))

            cmd_decode = [
                sys.executable,
                str(check_script),
                str(pred_llr_path),
                str(ldpc_cfg),
                str(pdsch_cfg),
                str(txbits),
                "--rmunmatch",
                str(rmunmatch),
                "--ldpctest",
                str(ldpctest),
                "--llr-scale",
                str(args.llr_scale),
                "--out",
                args.decoded_bits_name,
                "--meta",
                args.decode_meta_name,
            ]
            p_decode = run_cmd(cmd_decode, cwd=bdir)

            decode_meta = maybe_read_json(decode_meta_path)
            ber_pred = None
            bit_errors_pred = None
            rm_rc = None
            ldpc_rc = None
            if decode_meta is not None:
                ber_pred = decode_meta.get("ber", None)
                bit_errors_pred = decode_meta.get("bit_errors", None)
                rm_rc = decode_meta.get("rmunmatch_rc", None)
                ldpc_rc = decode_meta.get("ldpctest_rc", None)

            oai_baseline = None
            if args.also_run_oai_baseline:
                oai_meta_path = bdir / args.oai_decode_meta_name
                cmd_oai = [
                    sys.executable,
                    str(check_script),
                    str(demapper_llr),
                    str(ldpc_cfg),
                    str(pdsch_cfg),
                    str(txbits),
                    "--rmunmatch",
                    str(rmunmatch),
                    "--ldpctest",
                    str(ldpctest),
                    "--llr-scale",
                    "1.0",
                    "--out",
                    args.oai_decoded_bits_name,
                    "--meta",
                    args.oai_decode_meta_name,
                ]
                p_oai = run_cmd(cmd_oai, cwd=bdir)
                oai_meta = maybe_read_json(oai_meta_path)
                oai_baseline = {
                    "rc": p_oai.returncode,
                    "stdout_tail": safe_tail(p_oai.stdout),
                    "meta": oai_meta,
                }

            status = "OK"
            if p_decode.returncode != 0:
                status = "DECODE_SCRIPT_FAIL"
            elif decode_meta is None:
                status = "DECODE_META_MISSING"
            elif rm_rc not in (0, None) or ldpc_rc not in (0, None):
                status = "PIPE_RC_FAIL"

            one = {
                "bundle": bdir.name,
                "status": status,
                "pred_llr_path": str(pred_llr_path),
                "decode_meta_path": str(decode_meta_path),
                "infer_meta_path": str(infer_meta_path),
                "ber_pred": ber_pred,
                "bit_errors_pred": bit_errors_pred,
                "decode_script_rc": p_decode.returncode,
                "decode_stdout_tail": safe_tail(p_decode.stdout),
                "decode_meta": decode_meta,
                "oai_baseline": oai_baseline,
            }
            results.append(one)

            if status == "OK":
                processed += 1
                print(
                    f"[OK] {bdir.name}  "
                    f"BER_pred={ber_pred if ber_pred is not None else 'NA'}  "
                    f"bit_errors={bit_errors_pred if bit_errors_pred is not None else 'NA'}"
                )
            else:
                failed += 1
                print(f"[FAIL] {bdir.name}  status={status}")

        except Exception as e:
            failed += 1
            results.append({
                "bundle": bdir.name,
                "status": "EXCEPTION",
                "error": repr(e),
            })
            print(f"[EXCEPTION] {bdir.name}  {repr(e)}")

    summary = {
        "bundle_root": str(bundle_root),
        "ckpt": str(ckpt_path),
        "check_script": str(check_script),
        "rmunmatch": str(rmunmatch),
        "ldpctest": str(ldpctest),
        "llr_scale": args.llr_scale,
        "processed_ok": processed,
        "skipped": skipped,
        "failed": failed,
        "num_results": len(results),
        "results": results,
    }

    summary_path = bundle_root / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
