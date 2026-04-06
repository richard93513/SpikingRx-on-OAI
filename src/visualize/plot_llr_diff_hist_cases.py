#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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
) -> Path:
    summary_json = snapshot_dir / f"spikingrx_batch_summary_noise_power_{noise_label}.json"
    rows = read_spikingrx_summary_rows(summary_json)

    valid_items = [item for item in rows if is_valid_bundle(item)]
    if not valid_items:
        raise RuntimeError(f"no valid bundle for {noise_label}")

    selected_name = None

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
                selected_name = get_bundle_name(item)
                break

    if selected_name is None:
        selected_name = get_bundle_name(valid_items[0])

    bdir = snapshot_dir / f"bundle_noise_power_{noise_label}" / selected_name
    if not bdir.is_dir():
        raise RuntimeError(f"bundle dir not found: {bdir}")
    return bdir


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


def calc_diff_stats(spx: np.ndarray, oai: np.ndarray) -> Dict[str, float]:
    d = spx - oai
    abs_d = np.abs(d)

    return {
        "llr_len": int(d.size),
        "mean_diff": float(np.mean(d)),
        "std_diff": float(np.std(d)),
        "mean_abs_diff": float(np.mean(abs_d)),
        "median_abs_diff": float(np.median(abs_d)),
        "max_abs_diff": float(np.max(abs_d)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "p90_abs_diff": float(np.percentile(abs_d, 90)),
        "p95_abs_diff": float(np.percentile(abs_d, 95)),
        "p99_abs_diff": float(np.percentile(abs_d, 99)),
    }


def clip_range_from_percentile(d: np.ndarray, percentile: float) -> float:
    x = float(np.percentile(np.abs(d), percentile))
    return max(x, 1e-6)


def make_single_hist_plot(
    d: np.ndarray,
    title: str,
    out_png: Path,
    bins: int,
    clip_percentile: float,
    density: bool,
) -> Dict[str, float]:
    clip_abs = clip_range_from_percentile(d, clip_percentile)
    d_clip = np.clip(d, -clip_abs, clip_abs)

    plt.figure(figsize=(6.6, 5.4))
    plt.hist(d_clip, bins=bins, density=density)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("LLR difference = SpikingRx LLR - OAI LLR")
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "hist_clip_abs": clip_abs,
        "hist_min": -clip_abs,
        "hist_max": clip_abs,
    }


def make_combined_hist_plot(
    case_payloads: List[dict],
    out_png: Path,
    bins: int,
    clip_percentile: float,
    density: bool,
) -> None:
    n = len(case_payloads)
    plt.figure(figsize=(6.2 * n, 5.4))

    for i, case in enumerate(case_payloads, start=1):
        d = case["diff"]
        clip_abs = clip_range_from_percentile(d, clip_percentile)
        d_clip = np.clip(d, -clip_abs, clip_abs)

        plt.subplot(1, n, i)
        plt.hist(d_clip, bins=bins, density=density)
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("SpikingRx - OAI")
        plt.ylabel("Density" if density else "Count")
        plt.title(case["short_title"])
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def write_case_summary_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
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
        "p90_abs_diff",
        "p95_abs_diff",
        "p99_abs_diff",
        "hist_clip_abs",
        "single_plot_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    ap = argparse.ArgumentParser(
        description="Plot representative LLR-difference histograms: (SpikingRx LLR - OAI LLR)"
    )
    ap.add_argument("--snapshot_dir", type=str, default=str(DEFAULT_SNAPSHOT_DIR))
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--clip_percentile", type=float, default=99.5)
    ap.add_argument("--density", action="store_true", help="plot histogram as density instead of count")
    ap.add_argument(
        "--noise_labels",
        type=str,
        default="",
        help="comma-separated noise labels; empty means auto-select high/cliff/low cases",
    )
    args = ap.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not snapshot_dir.exists():
        raise SystemExit(f"[ERROR] snapshot dir not found: {snapshot_dir}")

    compare_csv = snapshot_dir / "compare_spikingrx_vs_oai_ber_vs_snr.csv"
    if not compare_csv.exists():
        raise SystemExit(f"[ERROR] missing {compare_csv}")

    compare_rows = read_compare_csv(compare_csv)

    if args.noise_labels.strip():
        noise_labels = [x.strip() for x in args.noise_labels.split(",") if x.strip()]
    else:
        noise_labels = choose_case_noise_labels(compare_rows)

    noise_to_snr_map = read_noise_to_snr_csv(snapshot_dir / "noise_to_snr_summary.csv")

    case_roles_default = ["high_snr", "cliff", "low_snr"]
    case_payloads = []
    summary_rows = []

    out_dir = snapshot_dir / "llr_diff_hist_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== LLR difference histogram representative cases ===")
    print(f"{'role':>10} | {'noise':>10} | {'snr_db':>10} | {'bundle':>34} | {'mad':>10} | {'p95':>10} | {'max':>10}")
    print("-" * 120)

    for i, noise_label in enumerate(noise_labels):
        role = case_roles_default[i] if i < len(case_roles_default) else f"case_{i+1}"

        if noise_label not in noise_to_snr_map:
            raise RuntimeError(f"noise label not found in noise_to_snr_summary.csv: {noise_label}")

        noise_db = noise_label_to_db(noise_label)
        snr_db = float(noise_to_snr_map[noise_label]["snr_top10_db"])

        bdir = choose_bundle_for_case(snapshot_dir, noise_label, pick_mode="first_nonzero_if_possible")
        spx, oai = load_llr_pair(bdir)
        d = spx - oai
        stats = calc_diff_stats(spx, oai)

        single_png = out_dir / f"llr_diff_hist_{role}_{noise_label}.png"
        title = (
            f"{role} | noise={format_db_value(noise_db)} dB | "
            f"SNR={snr_db:.3f} dB\n"
            f"{bdir.name}"
        )
        short_title = (
            f"{role}\nnoise={format_db_value(noise_db)} dB, SNR={snr_db:.3f} dB"
        )

        hist_info = make_single_hist_plot(
            d=d,
            title=title,
            out_png=single_png,
            bins=args.bins,
            clip_percentile=args.clip_percentile,
            density=args.density,
        )

        case_payloads.append({
            "role": role,
            "noise_label": noise_label,
            "noise_db": noise_db,
            "snr_top10_db": snr_db,
            "bundle_name": bdir.name,
            "diff": d,
            "short_title": short_title,
            "single_plot_path": str(single_png),
            "stats": stats,
            "hist_clip_abs": hist_info["hist_clip_abs"],
        })

        row = {
            "case_role": role,
            "noise_label": noise_label,
            "noise_db": noise_db,
            "snr_top10_db": snr_db,
            "bundle_name": bdir.name,
            "llr_len": stats["llr_len"],
            "mean_diff": stats["mean_diff"],
            "std_diff": stats["std_diff"],
            "mean_abs_diff": stats["mean_abs_diff"],
            "median_abs_diff": stats["median_abs_diff"],
            "max_abs_diff": stats["max_abs_diff"],
            "rmse": stats["rmse"],
            "p90_abs_diff": stats["p90_abs_diff"],
            "p95_abs_diff": stats["p95_abs_diff"],
            "p99_abs_diff": stats["p99_abs_diff"],
            "hist_clip_abs": hist_info["hist_clip_abs"],
            "single_plot_path": str(single_png),
        }
        summary_rows.append(row)

        print(
            f"{role:>10} | "
            f"{noise_label:>10} | "
            f"{snr_db:>10.6f} | "
            f"{bdir.name:>34} | "
            f"{stats['mean_abs_diff']:>10.6f} | "
            f"{stats['p95_abs_diff']:>10.6f} | "
            f"{stats['max_abs_diff']:>10.6f}"
        )

    combined_png = out_dir / "llr_diff_hist_cases_combined.png"
    make_combined_hist_plot(
        case_payloads=case_payloads,
        out_png=combined_png,
        bins=args.bins,
        clip_percentile=args.clip_percentile,
        density=args.density,
    )

    summary_csv = out_dir / "llr_diff_hist_cases_summary.csv"
    write_case_summary_csv(summary_csv, summary_rows)

    print("\n[OK] wrote:")
    print(f"  {combined_png}")
    print(f"  {summary_csv}")
    for row in summary_rows:
        print(f"  {row['single_plot_path']}")


if __name__ == "__main__":
    main()
