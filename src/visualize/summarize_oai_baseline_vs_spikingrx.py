#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 snapshots_snr 上做 OAI baseline：
  demapper_llr_f32.bin -> check_oai_llr_decode.py -> BER
再和既有的 SpikingRx BER vs SNR 做疊圖比較。

輸出：
  - oai_noise_sweep_summary.csv
  - oai_ber_vs_snr.csv
  - oai_ber_vs_snr.png
  - compare_spikingrx_vs_oai_ber_vs_snr.csv
  - compare_spikingrx_vs_oai_ber_vs_snr.png

設計重點
--------
1. snapshots_snr 內的每個 noise folder 共用同一個 snr_top10_run_summary.txt
2. bundle 的選取規則，跟既有 spikingrx_batch_summary_noise_power_*.json 對齊：
   - 先挑 status=OK 的 valid bundle
   - 再只取前 N 個 valid bundle
3. OAI baseline 只換 LLR 來源：
   - SpikingRx: predicted LLR -> LDPC -> BER
   - OAI     : demapper_llr_f32.bin -> LDPC -> BER
4. rm/unmatch 與 LDPC decode 完全沿用既有 check_oai_llr_decode.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = Path.home() / "SpikingRx-on-OAI"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "spx_records" / "snapshots_snr"
DEFAULT_CHECK_SCRIPT = REPO_ROOT / "src" / "inference" / "check_oai_llr_decode.py"
DEFAULT_RMUNMATCH = Path.home() / "openairinterface5g" / "cmake_targets" / "ran_build" / "build" / "rmunmatch_spx"
DEFAULT_LDPCTEST = Path.home() / "openairinterface5g" / "cmake_targets" / "ran_build" / "build" / "ldpctest_spx"

DEFAULT_TAKE_FIRST_VALID_N = 1500
TAKE_OVERRIDE: Dict[str, int] = {
    # "minus4.65dB": 1500,
}


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def safe_tail(s: str, n: int = 3000) -> str:
    if not s:
        return ""
    return s[-n:]


def parse_noise_label_from_summary_filename(name: str) -> str:
    m = re.match(r"spikingrx_batch_summary_noise_power_(.+)\.json$", name)
    if not m:
        raise ValueError(f"unexpected summary filename: {name}")
    return m.group(1)


def noise_label_to_db(label: str) -> float:
    m_pos = re.match(r"^(\d+(?:\.\d+)?)dB$", label)
    if m_pos:
        return float(m_pos.group(1))

    m_neg = re.match(r"^minus(\d+(?:\.\d+)?)dB$", label)
    if m_neg:
        return -float(m_neg.group(1))

    raise ValueError(f"cannot parse noise label: {label}")


def format_db_value(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    if abs(x * 10 - round(x * 10)) < 1e-9:
        return f"{x:.1f}"
    return f"{x:.2f}"


def find_bundle_list(summary_obj):
    if isinstance(summary_obj, dict):
        if "bundles" in summary_obj and isinstance(summary_obj["bundles"], list):
            return summary_obj["bundles"]
        if "results" in summary_obj and isinstance(summary_obj["results"], list):
            return summary_obj["results"]
        if "items" in summary_obj and isinstance(summary_obj["items"], list):
            return summary_obj["items"]
    raise RuntimeError("cannot find bundle list in summary json")


def get_bundle_name(item: dict) -> str:
    for k in ["bundle", "bundle_name", "bundle_dir", "name", "bdir"]:
        if k in item:
            return str(item[k])
    raise RuntimeError(f"cannot get bundle name from keys: {list(item.keys())}")


def get_status(item: dict) -> str:
    return str(item.get("status", "")).strip().lower()


def get_decode_script_rc(item: dict):
    if "decode_script_rc" not in item:
        return None
    try:
        return int(item["decode_script_rc"])
    except Exception:
        return None


def has_required_pred_fields(item: dict) -> bool:
    has_err = ("bit_errors_pred" in item) or ("bit_errors" in item) or ("errors" in item)
    has_ber = ("ber_pred" in item) or ("ber" in item)
    return has_err and has_ber


def is_valid_bundle(item: dict) -> bool:
    status = get_status(item)
    if status not in ("ok",):
        return False

    rc = get_decode_script_rc(item)
    if rc is not None and rc != 0:
        return False

    if not has_required_pred_fields(item):
        return False

    return True


def read_spikingrx_summary_rows(summary_json: Path) -> List[dict]:
    obj = json.loads(summary_json.read_text())
    return find_bundle_list(obj)


def list_valid_bundle_dirs_for_noise(snapshot_dir: Path, noise_label: str) -> List[Path]:
    summary_json = snapshot_dir / f"spikingrx_batch_summary_noise_power_{noise_label}.json"
    rows = read_spikingrx_summary_rows(summary_json)
    take_n = TAKE_OVERRIDE.get(noise_label, DEFAULT_TAKE_FIRST_VALID_N)

    valid_names: List[str] = []
    for item in rows:
        if is_valid_bundle(item):
            valid_names.append(get_bundle_name(item))
            if len(valid_names) >= take_n:
                break

    bundle_root = snapshot_dir / f"bundle_noise_power_{noise_label}"
    out = []
    for name in valid_names:
        bdir = bundle_root / name
        if bdir.is_dir():
            out.append(bdir)
    return out


def read_noise_to_snr_csv(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            noise_label = row["noise_label"]
            rec = dict(row)
            for k in ["noise_db", "snr_top10_db", "used_bundle_count", "snr_used_windows"]:
                if k in rec and rec[k] != "":
                    try:
                        if k in ("used_bundle_count", "snr_used_windows"):
                            rec[k] = int(float(rec[k]))
                        else:
                            rec[k] = float(rec[k])
                    except Exception:
                        pass
            out[noise_label] = rec
    return out


def read_existing_spikingrx_ber_vs_snr(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["noise_label"]] = dict(row)
    return out


def read_meta_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def decode_one_oai_bundle(
    bdir: Path,
    check_script: Path,
    rmunmatch: Path,
    ldpctest: Path,
    llr_scale: float,
    out_name: str,
    meta_name: str,
    force_rerun: bool,
) -> dict:
    meta_path = bdir / meta_name

    if (not force_rerun) and meta_path.exists():
        meta = read_meta_json(meta_path)
        if meta is not None:
            return {
                "bundle": bdir.name,
                "status": "CACHED",
                "meta": meta,
                "stdout_tail": "",
            }

    llr_bin = bdir / "demapper_llr_f32.bin"
    ldpc_cfg = bdir / "ldpc_cfg.txt"
    pdsch_cfg = bdir / "pdsch_cfg.txt"
    txbits = bdir / "txbits.bin"

    required = [llr_bin, ldpc_cfg, pdsch_cfg, txbits]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return {
            "bundle": bdir.name,
            "status": "MISSING_FILES",
            "missing": missing,
            "meta": None,
            "stdout_tail": "",
        }

    cmd = [
        sys.executable,
        str(check_script),
        str(llr_bin),
        str(ldpc_cfg),
        str(pdsch_cfg),
        str(txbits),
        "--rmunmatch",
        str(rmunmatch),
        "--ldpctest",
        str(ldpctest),
        "--llr-scale",
        str(llr_scale),
        "--out",
        out_name,
        "--meta",
        meta_name,
    ]
    p = run_cmd(cmd, cwd=bdir)
    meta = read_meta_json(meta_path)

    status = "OK"
    if p.returncode != 0:
        status = "DECODE_SCRIPT_FAIL"
    elif meta is None:
        status = "META_MISSING"
    else:
        rm_rc = meta.get("rmunmatch_rc", None)
        ldpc_rc = meta.get("ldpctest_rc", None)
        if rm_rc not in (0, None) or ldpc_rc not in (0, None):
            status = "PIPE_RC_FAIL"

    return {
        "bundle": bdir.name,
        "status": status,
        "meta": meta,
        "stdout_tail": safe_tail(p.stdout),
    }


def summarize_one_noise(
    snapshot_dir: Path,
    noise_label: str,
    snr_row: Optional[dict],
    check_script: Path,
    rmunmatch: Path,
    ldpctest: Path,
    llr_scale: float,
    out_name: str,
    meta_name: str,
    force_rerun: bool,
) -> Tuple[dict, List[dict]]:
    noise_db = noise_label_to_db(noise_label)
    bdirs = list_valid_bundle_dirs_for_noise(snapshot_dir, noise_label)

    bundle_results: List[dict] = []
    total_bit_errors = 0
    total_bits = 0
    nonzero_bundle_count = 0
    ok_count = 0
    fail_count = 0

    first_nonzero_bundle = None
    worst_bundle_name = None
    worst_bundle_ber = -1.0

    for bdir in bdirs:
        one = decode_one_oai_bundle(
            bdir=bdir,
            check_script=check_script,
            rmunmatch=rmunmatch,
            ldpctest=ldpctest,
            llr_scale=llr_scale,
            out_name=out_name,
            meta_name=meta_name,
            force_rerun=force_rerun,
        )

        meta = one.get("meta")
        if one["status"] == "OK" or one["status"] == "CACHED":
            ok_count += 1
            if meta is None:
                fail_count += 1
                one["status"] = "META_MISSING"
            else:
                ber = float(meta.get("ber", math.nan))
                bit_errors = int(meta.get("bit_errors", 0))
                A = int(meta.get("A", 0))

                total_bit_errors += bit_errors
                total_bits += A

                if bit_errors > 0:
                    nonzero_bundle_count += 1
                    if first_nonzero_bundle is None:
                        first_nonzero_bundle = {
                            "name": bdir.name,
                            "bit_errors": bit_errors,
                            "bits": A,
                            "ber": ber,
                        }

                if math.isfinite(ber) and ber > worst_bundle_ber:
                    worst_bundle_ber = ber
                    worst_bundle_name = bdir.name
        else:
            fail_count += 1

        bundle_results.append(one)

    group_ber = (total_bit_errors / total_bits) if total_bits > 0 else math.nan

    row = {
        "noise_label": noise_label,
        "noise_db": noise_db,
        "snr_top10_db": float(snr_row.get("snr_top10_db", math.nan)) if snr_row else math.nan,
        "take_n": TAKE_OVERRIDE.get(noise_label, DEFAULT_TAKE_FIRST_VALID_N),
        "selected_bundle_count": len(bdirs),
        "oai_ok_bundle_count": ok_count,
        "oai_fail_bundle_count": fail_count,
        "nonzero_bundle_count_oai": nonzero_bundle_count,
        "total_bit_errors_oai": total_bit_errors,
        "total_bits_oai": total_bits,
        "group_ber_oai": group_ber,
        "worst_bundle_name": worst_bundle_name,
        "worst_bundle_ber_oai": worst_bundle_ber,
        "first_nonzero_bundle_name": first_nonzero_bundle["name"] if first_nonzero_bundle else "",
        "first_nonzero_bundle_bit_errors_oai": first_nonzero_bundle["bit_errors"] if first_nonzero_bundle else "",
        "first_nonzero_bundle_bits": first_nonzero_bundle["bits"] if first_nonzero_bundle else "",
        "first_nonzero_bundle_ber_oai": first_nonzero_bundle["ber"] if first_nonzero_bundle else "",
    }
    return row, bundle_results


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_single_curve(x: List[float], y: List[float], out_png: Path, title: str, xlabel: str, ylabel: str) -> None:
    x_ok, y_ok = [], []
    for xx, yy in zip(x, y):
        if math.isfinite(xx) and math.isfinite(yy):
            x_ok.append(xx)
            y_ok.append(yy if yy > 0.0 else 1e-12)

    if not x_ok:
        return

    plt.figure(figsize=(8, 5))
    plt.semilogy(x_ok, y_ok, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_compare(spx_rows: List[dict], oai_rows: List[dict], out_png: Path) -> None:
    spx_map = {r["noise_label"]: r for r in spx_rows}
    oai_map = {r["noise_label"]: r for r in oai_rows}
    common = sorted(set(spx_map.keys()) & set(oai_map.keys()), key=lambda k: noise_label_to_db(k))
    if not common:
        return

    x_spx, y_spx = [], []
    x_oai, y_oai = [], []
    for k in common:
        rs = spx_map[k]
        ro = oai_map[k]
        xs = float(rs["snr_top10_db"])
        ys = float(rs["group_ber_pred"])
        xo = float(ro["snr_top10_db"])
        yo = float(ro["group_ber_oai"])
        if math.isfinite(xs) and math.isfinite(ys):
            x_spx.append(xs)
            y_spx.append(ys if ys > 0.0 else 1e-12)
        if math.isfinite(xo) and math.isfinite(yo):
            x_oai.append(xo)
            y_oai.append(yo if yo > 0.0 else 1e-12)

    if not x_spx and not x_oai:
        return

    plt.figure(figsize=(8, 5))
    if x_spx:
        plt.semilogy(x_spx, y_spx, marker="o", label="SpikingRx")
    if x_oai:
        plt.semilogy(x_oai, y_oai, marker="s", label="OAI demapper baseline")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Group BER")
    plt.title("SpikingRx vs OAI BER vs SNR")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def build_spikingrx_rows_for_compare(
    noise_to_snr_map: Dict[str, dict],
    spikingrx_ber_vs_snr_path: Path,
    fallback_noise_sweep_path: Path,
) -> List[dict]:
    rows: List[dict] = []

    if spikingrx_ber_vs_snr_path.exists():
        with spikingrx_ber_vs_snr_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                noise_label = row["noise_label"]
                snr = float(row["snr_top10_db"]) if row.get("snr_top10_db", "") != "" else float(noise_to_snr_map[noise_label]["snr_top10_db"])
                ber = float(row["group_ber_pred"])
                rows.append({
                    "noise_label": noise_label,
                    "noise_db": float(row["noise_db"]),
                    "snr_top10_db": snr,
                    "group_ber_pred": ber,
                })
        return rows

    if fallback_noise_sweep_path.exists():
        with fallback_noise_sweep_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                noise_label = row["noise_label"]
                if noise_label not in noise_to_snr_map:
                    continue
                rows.append({
                    "noise_label": noise_label,
                    "noise_db": float(row["noise_db"]),
                    "snr_top10_db": float(noise_to_snr_map[noise_label]["snr_top10_db"]),
                    "group_ber_pred": float(row["group_ber_pred"]),
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description="OAI baseline BER vs SNR on snapshots_snr")
    ap.add_argument("--snapshot_dir", type=str, default=str(DEFAULT_SNAPSHOT_DIR))
    ap.add_argument("--check_script", type=str, default=str(DEFAULT_CHECK_SCRIPT))
    ap.add_argument("--rmunmatch", type=str, default=str(DEFAULT_RMUNMATCH))
    ap.add_argument("--ldpctest", type=str, default=str(DEFAULT_LDPCTEST))
    ap.add_argument("--llr_scale", type=float, default=1.0)
    ap.add_argument("--oai_out_name", type=str, default="decoded_bits_oai.bin")
    ap.add_argument("--oai_meta_name", type=str, default="oai_decode_meta.json")
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    check_script = Path(args.check_script).resolve()
    rmunmatch = Path(args.rmunmatch).resolve()
    ldpctest = Path(args.ldpctest).resolve()

    if not snapshot_dir.exists():
        raise SystemExit(f"[ERROR] snapshot_dir not found: {snapshot_dir}")
    if not check_script.exists():
        raise SystemExit(f"[ERROR] check_script not found: {check_script}")
    if not rmunmatch.exists():
        raise SystemExit(f"[ERROR] rmunmatch not found: {rmunmatch}")
    if not ldpctest.exists():
        raise SystemExit(f"[ERROR] ldpctest not found: {ldpctest}")

    noise_to_snr_csv = snapshot_dir / "noise_to_snr_summary.csv"
    if not noise_to_snr_csv.exists():
        raise SystemExit(f"[ERROR] missing {noise_to_snr_csv}")
    noise_to_snr_map = read_noise_to_snr_csv(noise_to_snr_csv)

    summary_jsons = sorted(
        snapshot_dir.glob("spikingrx_batch_summary_noise_power_*.json"),
        key=lambda p: noise_label_to_db(parse_noise_label_from_summary_filename(p.name)),
    )
    if not summary_jsons:
        raise SystemExit(f"[ERROR] no summary json found in: {snapshot_dir}")

    oai_rows: List[dict] = []
    detail_json: Dict[str, List[dict]] = {}

    print("\n=== OAI baseline decode ===")
    for jpath in summary_jsons:
        noise_label = parse_noise_label_from_summary_filename(jpath.name)
        snr_row = noise_to_snr_map.get(noise_label)

        row, bundle_results = summarize_one_noise(
            snapshot_dir=snapshot_dir,
            noise_label=noise_label,
            snr_row=snr_row,
            check_script=check_script,
            rmunmatch=rmunmatch,
            ldpctest=ldpctest,
            llr_scale=args.llr_scale,
            out_name=args.oai_out_name,
            meta_name=args.oai_meta_name,
            force_rerun=args.force_rerun,
        )
        oai_rows.append(row)
        detail_json[noise_label] = bundle_results

        snr_s = "nan" if not math.isfinite(row["snr_top10_db"]) else f"{row['snr_top10_db']:.6f}"
        ber_s = "nan" if not math.isfinite(row["group_ber_oai"]) else f"{row['group_ber_oai']:.6e}"
        print(
            f"noise={format_db_value(row['noise_db']):>7} dB | "
            f"snr={snr_s:>10} | "
            f"used={row['selected_bundle_count']:>4} | "
            f"ok={row['oai_ok_bundle_count']:>4} | "
            f"err={row['total_bit_errors_oai']:>8} | "
            f"BER_oai={ber_s}"
        )

    oai_rows.sort(key=lambda r: r["noise_db"])

    oai_noise_sweep_csv = snapshot_dir / "oai_noise_sweep_summary.csv"
    write_csv(
        oai_noise_sweep_csv,
        oai_rows,
        fieldnames=[
            "noise_db",
            "noise_label",
            "snr_top10_db",
            "take_n",
            "selected_bundle_count",
            "oai_ok_bundle_count",
            "oai_fail_bundle_count",
            "nonzero_bundle_count_oai",
            "total_bit_errors_oai",
            "total_bits_oai",
            "group_ber_oai",
            "worst_bundle_name",
            "worst_bundle_ber_oai",
            "first_nonzero_bundle_name",
            "first_nonzero_bundle_bit_errors_oai",
            "first_nonzero_bundle_bits",
            "first_nonzero_bundle_ber_oai",
        ],
    )

    oai_ber_vs_snr_rows = [
        {
            "noise_db": r["noise_db"],
            "noise_label": r["noise_label"],
            "snr_top10_db": r["snr_top10_db"],
            "group_ber_oai": r["group_ber_oai"],
            "selected_bundle_count": r["selected_bundle_count"],
            "oai_ok_bundle_count": r["oai_ok_bundle_count"],
            "oai_fail_bundle_count": r["oai_fail_bundle_count"],
            "total_bit_errors_oai": r["total_bit_errors_oai"],
            "total_bits_oai": r["total_bits_oai"],
        }
        for r in sorted(oai_rows, key=lambda x: (x["snr_top10_db"] if math.isfinite(x["snr_top10_db"]) else -1e18))
    ]

    oai_ber_vs_snr_csv = snapshot_dir / "oai_ber_vs_snr.csv"
    write_csv(
        oai_ber_vs_snr_csv,
        oai_ber_vs_snr_rows,
        fieldnames=[
            "noise_db",
            "noise_label",
            "snr_top10_db",
            "group_ber_oai",
            "selected_bundle_count",
            "oai_ok_bundle_count",
            "oai_fail_bundle_count",
            "total_bit_errors_oai",
            "total_bits_oai",
        ],
    )

    oai_ber_vs_snr_png = snapshot_dir / "oai_ber_vs_snr.png"
    plot_single_curve(
        x=[r["snr_top10_db"] for r in oai_ber_vs_snr_rows],
        y=[r["group_ber_oai"] for r in oai_ber_vs_snr_rows],
        out_png=oai_ber_vs_snr_png,
        title="OAI demapper baseline BER vs SNR",
        xlabel="SNR (dB)",
        ylabel="Group BER",
    )

    spikingrx_rows = build_spikingrx_rows_for_compare(
        noise_to_snr_map=noise_to_snr_map,
        spikingrx_ber_vs_snr_path=snapshot_dir / "ber_vs_snr.csv",
        fallback_noise_sweep_path=snapshot_dir / "noise_sweep_summary.csv",
    )

    compare_rows: List[dict] = []
    spx_map = {r["noise_label"]: r for r in spikingrx_rows}
    oai_map = {r["noise_label"]: r for r in oai_rows}
    common = sorted(set(spx_map.keys()) & set(oai_map.keys()), key=lambda k: noise_label_to_db(k))
    for k in common:
        rs = spx_map[k]
        ro = oai_map[k]
        compare_rows.append({
            "noise_db": ro["noise_db"],
            "noise_label": k,
            "snr_top10_db": ro["snr_top10_db"],
            "group_ber_spikingrx": rs["group_ber_pred"],
            "group_ber_oai": ro["group_ber_oai"],
            "spikingrx_over_oai_ratio": (
                (float(rs["group_ber_pred"]) / float(ro["group_ber_oai"]))
                if float(ro["group_ber_oai"]) > 0.0 else ""
            ),
        })

    compare_csv = snapshot_dir / "compare_spikingrx_vs_oai_ber_vs_snr.csv"
    write_csv(
        compare_csv,
        compare_rows,
        fieldnames=[
            "noise_db",
            "noise_label",
            "snr_top10_db",
            "group_ber_spikingrx",
            "group_ber_oai",
            "spikingrx_over_oai_ratio",
        ],
    )

    compare_png = snapshot_dir / "compare_spikingrx_vs_oai_ber_vs_snr.png"
    plot_compare(spikingrx_rows, oai_rows, compare_png)

    detail_json_path = snapshot_dir / "oai_decode_detail.json"
    detail_json_path.write_text(json.dumps(detail_json, indent=2, ensure_ascii=False))

    print("\n[OK] wrote:")
    print(f"  {oai_noise_sweep_csv}")
    print(f"  {oai_ber_vs_snr_csv}")
    print(f"  {oai_ber_vs_snr_png}")
    print(f"  {compare_csv}")
    print(f"  {compare_png}")
    print(f"  {detail_json_path}")


if __name__ == "__main__":
    main()
