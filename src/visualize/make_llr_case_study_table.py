#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path.home() / "SpikingRx-on-OAI"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "spx_records" / "snapshots_snr"

DEFAULT_TAKE_FIRST_VALID_N = 1500
TAKE_OVERRIDE: Dict[str, int] = {
    # "minus4.65dB": 1500,
}


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


def read_noise_to_snr_csv(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["noise_label"]] = {
                "noise_label": row["noise_label"],
                "noise_db": float(row["noise_db_from_name"]) if "noise_db_from_name" in row else float(row["noise_db"]),
                "snr_top10_db": float(row["snr_top10_db"]),
            }
    return out


def read_compare_csv(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "noise_label": row["noise_label"],
                "noise_db": float(row["noise_db"]),
                "snr_top10_db": float(row["snr_top10_db"]),
                "group_ber_spikingrx": float(row["group_ber_spikingrx"]),
                "group_ber_oai": float(row["group_ber_oai"]),
            })
    return rows


def read_spikingrx_summary_rows(summary_json: Path) -> List[dict]:
    obj = json.loads(summary_json.read_text())
    return find_bundle_list(obj)


def choose_case_noise_labels(compare_rows: List[dict]) -> List[str]:
    if not compare_rows:
        raise RuntimeError("compare csv is empty")

    rows_sorted_snr = sorted(compare_rows, key=lambda r: r["snr_top10_db"])
    low_noise_label = rows_sorted_snr[0]["noise_label"]
    high_noise_label = rows_sorted_snr[-1]["noise_label"]

    positive_rows = [r for r in compare_rows if r["group_ber_spikingrx"] > 0.0]
    if positive_rows:
        cliff_row = min(positive_rows, key=lambda r: r["group_ber_spikingrx"])
    else:
        cliff_row = min(compare_rows, key=lambda r: abs(r["group_ber_spikingrx"] - 1e-3))

    cliff_noise_label = cliff_row["noise_label"]

    out = []
    for x in [high_noise_label, cliff_noise_label, low_noise_label]:
        if x not in out:
            out.append(x)
    return out


def choose_bundle_for_case(
    snapshot_dir: Path,
    noise_label: str,
    pick_mode: str = "first_nonzero_if_possible",
) -> Tuple[Path, dict]:
    summary_json = snapshot_dir / f"spikingrx_batch_summary_noise_power_{noise_label}.json"
    rows = read_spikingrx_summary_rows(summary_json)

    valid_items = [item for item in rows if is_valid_bundle(item)]
    if not valid_items:
        raise RuntimeError(f"no valid bundle for {noise_label}")

    selected_item = None

    if pick_mode == "first_nonzero_if_possible":
        for item in valid_items:
            be = None
            if "bit_errors_pred" in item:
                be = int(item["bit_errors_pred"])
            elif "bit_errors" in item:
                be = int(item["bit_errors"])
            elif "errors" in item:
                be = int(item["errors"])
            if be is not None and be > 0:
                selected_item = item
                break

    if selected_item is None:
        selected_item = valid_items[0]

    bname = get_bundle_name(selected_item)
    bdir = snapshot_dir / f"bundle_noise_power_{noise_label}" / bname
    if not bdir.is_dir():
        raise RuntimeError(f"bundle dir not found: {bdir}")
    return bdir, selected_item


def load_llr_pair(bdir: Path) -> Tuple[np.ndarray, np.ndarray]:
    spx_path = bdir / "spikingrx_llr_f32.bin"
    oai_path = bdir / "demapper_llr_f32.bin"

    if not spx_path.exists():
        raise FileNotFoundError(f"missing {spx_path}")
    if not oai_path.exists():
        raise FileNotFoundError(f"missing {oai_path}")

    spx = np.fromfile(spx_path, dtype=np.float32)
    oai = np.fromfile(oai_path, dtype=np.float32)

    if spx.size != oai.size:
        raise RuntimeError(f"LLR length mismatch in {bdir.name}: spx={spx.size}, oai={oai.size}")
    if spx.size == 0:
        raise RuntimeError(f"empty llr files in {bdir.name}")

    return spx, oai


def read_meta_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def calc_pair_metrics(spx: np.ndarray, oai: np.ndarray) -> Dict[str, float]:
    d = spx - oai
    abs_d = np.abs(d)

    spx_sign = spx >= 0.0
    oai_sign = oai >= 0.0

    corr = float(np.corrcoef(spx, oai)[0, 1]) if spx.size >= 2 else float("nan")

    return {
        "llr_len": int(spx.size),
        "mean_diff": float(np.mean(d)),
        "std_diff": float(np.std(d)),
        "mean_abs_diff": float(np.mean(abs_d)),
        "median_abs_diff": float(np.median(abs_d)),
        "max_abs_diff": float(np.max(abs_d)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "corr": corr,
        "sign_agreement": float(np.mean(spx_sign == oai_sign)),
    }


def decode_row_from_meta(meta: Optional[dict]) -> Dict[str, object]:
    if meta is None:
        return {
            "ber": math.nan,
            "bit_errors": None,
            "A": None,
            "rmunmatch_rc": None,
            "ldpctest_rc": None,
            "llr_bin": "",
        }

    return {
        "ber": float(meta.get("ber", math.nan)),
        "bit_errors": int(meta["bit_errors"]) if "bit_errors" in meta and meta["bit_errors"] is not None else None,
        "A": int(meta["A"]) if "A" in meta and meta["A"] is not None else None,
        "rmunmatch_rc": meta.get("rmunmatch_rc", None),
        "ldpctest_rc": meta.get("ldpctest_rc", None),
        "llr_bin": meta.get("paths", {}).get("llr_bin", ""),
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_markdown_table(path: Path, rows: List[dict]) -> None:
    headers = [
        "Case",
        "noise_label",
        "SNR(dB)",
        "bundle",
        "mean_abs_diff",
        "corr",
        "sign_agree",
        "BER_spikingrx",
        "BER_oai",
        "bit_errors_spikingrx",
        "bit_errors_oai",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        line = [
            str(r["case_role"]),
            str(r["noise_label"]),
            f"{float(r['snr_top10_db']):.6f}" if r["snr_top10_db"] != "" else "",
            str(r["bundle_name"]),
            f"{float(r['mean_abs_diff']):.6f}" if r["mean_abs_diff"] != "" else "",
            f"{float(r['corr']):.6f}" if r["corr"] != "" else "",
            f"{float(r['sign_agreement']):.6f}" if r["sign_agreement"] != "" else "",
            f"{float(r['ber_spikingrx']):.12e}" if r["ber_spikingrx"] != "" and math.isfinite(float(r["ber_spikingrx"])) else "nan",
            f"{float(r['ber_oai']):.12e}" if r["ber_oai"] != "" and math.isfinite(float(r["ber_oai"])) else "nan",
            str(r["bit_errors_spikingrx"]),
            str(r["bit_errors_oai"]),
        ]
        lines.append("| " + " | ".join(line) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="Build representative case-study table for high-SNR / cliff / low-SNR bundles"
    )
    ap.add_argument("--snapshot_dir", type=str, default=str(DEFAULT_SNAPSHOT_DIR))
    ap.add_argument(
        "--noise_labels",
        type=str,
        default="",
        help="comma-separated noise labels; empty means auto-select high/cliff/low",
    )
    args = ap.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not snapshot_dir.exists():
        raise SystemExit(f"[ERROR] snapshot dir not found: {snapshot_dir}")

    compare_csv = snapshot_dir / "compare_spikingrx_vs_oai_ber_vs_snr.csv"
    if not compare_csv.exists():
        raise SystemExit(f"[ERROR] missing {compare_csv}")

    compare_rows = read_compare_csv(compare_csv)
    noise_to_snr_map = read_noise_to_snr_csv(snapshot_dir / "noise_to_snr_summary.csv")

    if args.noise_labels.strip():
        noise_labels = [x.strip() for x in args.noise_labels.split(",") if x.strip()]
    else:
        noise_labels = choose_case_noise_labels(compare_rows)

    case_roles_default = ["high_snr", "cliff", "low_snr"]
    rows_out = []

    out_dir = snapshot_dir / "llr_case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Representative LLR case-study table ===")
    print(
        f"{'case':>10} | {'noise':>10} | {'snr_db':>10} | {'bundle':>34} | "
        f"{'mad':>10} | {'corr':>10} | {'sign':>10} | {'ber_spx':>12} | {'ber_oai':>12}"
    )
    print("-" * 136)

    for i, noise_label in enumerate(noise_labels):
        role = case_roles_default[i] if i < len(case_roles_default) else f"case_{i+1}"

        if noise_label not in noise_to_snr_map:
            raise RuntimeError(f"noise label not found in noise_to_snr_summary.csv: {noise_label}")

        snr_db = float(noise_to_snr_map[noise_label]["snr_top10_db"])
        noise_db = noise_label_to_db(noise_label)

        bdir, summary_item = choose_bundle_for_case(
            snapshot_dir=snapshot_dir,
            noise_label=noise_label,
            pick_mode="first_nonzero_if_possible",
        )

        spx, oai = load_llr_pair(bdir)
        pair_metrics = calc_pair_metrics(spx, oai)

        spx_meta = read_meta_json(bdir / "spikingrx_decode_meta.json")
        oai_meta = read_meta_json(bdir / "oai_decode_meta_batch.json")
        spx_dec = decode_row_from_meta(spx_meta)
        oai_dec = decode_row_from_meta(oai_meta)

        row = {
            "case_role": role,
            "noise_label": noise_label,
            "noise_db": noise_db,
            "snr_top10_db": snr_db,
            "bundle_name": bdir.name,

            "llr_len": pair_metrics["llr_len"],
            "mean_diff": pair_metrics["mean_diff"],
            "std_diff": pair_metrics["std_diff"],
            "mean_abs_diff": pair_metrics["mean_abs_diff"],
            "median_abs_diff": pair_metrics["median_abs_diff"],
            "max_abs_diff": pair_metrics["max_abs_diff"],
            "rmse": pair_metrics["rmse"],
            "corr": pair_metrics["corr"],
            "sign_agreement": pair_metrics["sign_agreement"],

            "ber_spikingrx": spx_dec["ber"],
            "ber_oai": oai_dec["ber"],
            "bit_errors_spikingrx": spx_dec["bit_errors"],
            "bit_errors_oai": oai_dec["bit_errors"],
            "A_spikingrx": spx_dec["A"],
            "A_oai": oai_dec["A"],
            "rmunmatch_rc_spikingrx": spx_dec["rmunmatch_rc"],
            "ldpctest_rc_spikingrx": spx_dec["ldpctest_rc"],
            "rmunmatch_rc_oai": oai_dec["rmunmatch_rc"],
            "ldpctest_rc_oai": oai_dec["ldpctest_rc"],
            "llr_bin_spikingrx": spx_dec["llr_bin"],
            "llr_bin_oai": oai_dec["llr_bin"],

            "summary_item_status": summary_item.get("status", ""),
            "summary_item_ber_pred": summary_item.get("ber_pred", summary_item.get("ber", "")),
            "summary_item_bit_errors_pred": summary_item.get("bit_errors_pred", summary_item.get("bit_errors", summary_item.get("errors", ""))),
        }
        rows_out.append(row)

        ber_spx_s = "nan" if not math.isfinite(float(row["ber_spikingrx"])) else f"{float(row['ber_spikingrx']):.6e}"
        ber_oai_s = "nan" if not math.isfinite(float(row["ber_oai"])) else f"{float(row['ber_oai']):.6e}"

        print(
            f"{role:>10} | "
            f"{noise_label:>10} | "
            f"{snr_db:>10.6f} | "
            f"{bdir.name:>34} | "
            f"{row['mean_abs_diff']:>10.6f} | "
            f"{row['corr']:>10.6f} | "
            f"{row['sign_agreement']:>10.6f} | "
            f"{ber_spx_s:>12} | "
            f"{ber_oai_s:>12}"
        )

    csv_path = out_dir / "llr_case_study_table.csv"
    write_csv(
        csv_path,
        rows_out,
        fieldnames=[
            "case_role",
            "noise_label",
            "noise_db",
            "snr_top10_db",
            "bundle_name",
            "llr_len",
            "mean_diff",
            "std_diff",
            "mean_abs_diff",
            "median_abs_diff",
            "max_abs_diff",
            "rmse",
            "corr",
            "sign_agreement",
            "ber_spikingrx",
            "ber_oai",
            "bit_errors_spikingrx",
            "bit_errors_oai",
            "A_spikingrx",
            "A_oai",
            "rmunmatch_rc_spikingrx",
            "ldpctest_rc_spikingrx",
            "rmunmatch_rc_oai",
            "ldpctest_rc_oai",
            "llr_bin_spikingrx",
            "llr_bin_oai",
            "summary_item_status",
            "summary_item_ber_pred",
            "summary_item_bit_errors_pred",
        ],
    )

    md_path = out_dir / "llr_case_study_table.md"
    write_markdown_table(md_path, rows_out)

    print("\n[OK] wrote:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
