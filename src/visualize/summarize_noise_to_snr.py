#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpikingRx-on-OAI
LLR-based reliability proxy summary

Important
---------
This script does NOT compute physical SNR in dB.

Instead, it computes an LLR-domain reliability proxy from
demapper_llr_f32.bin, then joins it with BER summary produced by
summarize_noise_sweep.py.

Why:
- Current DMRS/fullgrid-based "SNR" attempts were not stable/reliable
  on the dumped OAI artifacts.
- BER is ultimately governed by soft-bit reliability.
- Therefore, for plotting and monotonic comparison, an LLR-based proxy
  is a better intermediate metric.

Outputs
-------
1) noise_to_snr_summary.csv
   (historical filename kept for compatibility, but content is proxy-based)
2) ber_vs_llr_proxy.csv
3) ber_vs_llr_proxy.png

Suggested interpretation
------------------------
Use this as:
    BER vs LLR reliability proxy

NOT as:
    BER vs physical SNR(dB)
"""

from pathlib import Path
import csv
import json
import math
import re
from statistics import median

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path.home() / "SpikingRx-on-OAI"
SNAPSHOT_DIR = REPO_ROOT / "spx_records" / "snapshots"

DEFAULT_TAKE_FIRST_VALID_N = 1500

TAKE_OVERRIDE = {
    # "0dB": 1500,
    # "2dB": 1500,
    # "minus1dB": 1500,
    # "minus2dB": 1500,
    # "minus3dB": 1500,
    # "minus3.5dB": 1500,
    # "minus4dB": 1500,
    # "minus4.3dB": 1500,
    # "minus4.5dB": 1500,
    # "minus4.7dB": 1500,
    # "minus4.8dB": 1500,
    # "minus4.9dB": 1500,
    # "minus5dB": 1500,
    # "minus6dB": 1500,
    # "minus8dB": 1500,
    # "minus10dB": 1500,
}

EPS = 1e-12


def parse_noise_label_from_filename(name: str) -> str:
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
    return f"{x:.1f}"


def json_load(path: Path):
    return json.loads(path.read_text())


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
    has_err = (
        "bit_errors_pred" in item
        or "bit_errors" in item
        or "errors" in item
    )
    has_ber = ("ber_pred" in item or "ber" in item)
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


def list_valid_bundle_dirs_for_noise(noise_label: str):
    summary_json = SNAPSHOT_DIR / f"spikingrx_batch_summary_noise_power_{noise_label}.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"summary json not found: {summary_json}")

    obj = json_load(summary_json)
    bundles = find_bundle_list(obj)

    take_n = TAKE_OVERRIDE.get(noise_label, DEFAULT_TAKE_FIRST_VALID_N)
    valid_names = []

    for item in bundles:
        if is_valid_bundle(item):
            valid_names.append(get_bundle_name(item))
            if len(valid_names) >= take_n:
                break

    bundle_root = SNAPSHOT_DIR / f"bundle_noise_power_{noise_label}"
    out = []
    for bname in valid_names:
        bdir = bundle_root / bname
        if bdir.is_dir():
            out.append(bdir)

    return out


def compute_llr_proxy_for_bundle(bdir: Path):
    llr_path = bdir / "demapper_llr_f32.bin"
    if not llr_path.exists():
        raise FileNotFoundError(f"missing {llr_path}")

    llr = np.fromfile(llr_path, dtype=np.float32)
    if llr.size == 0:
        raise RuntimeError(f"empty llr file: {llr_path}")

    abs_llr = np.abs(llr)

    return {
        "bundle": bdir.name,
        "count": int(llr.size),
        "llr_mean": float(np.mean(llr)),
        "llr_std": float(np.std(llr)),
        "llr_abs_mean": float(np.mean(abs_llr)),
        "llr_abs_rms": float(np.sqrt(np.mean(abs_llr ** 2))),
        "llr_abs_median": float(np.median(abs_llr)),
        "llr_abs_p75": float(np.percentile(abs_llr, 75)),
        "llr_abs_p90": float(np.percentile(abs_llr, 90)),
        "llr_abs_p95": float(np.percentile(abs_llr, 95)),
        "llr_abs_p99": float(np.percentile(abs_llr, 99)),
        "llr_abs_min": float(np.min(abs_llr)),
        "llr_abs_max": float(np.max(abs_llr)),
    }


def summarize_noise_to_proxy():
    summary_jsons = sorted(
        SNAPSHOT_DIR.glob("spikingrx_batch_summary_noise_power_*.json"),
        key=lambda p: noise_label_to_db(parse_noise_label_from_filename(p.name)),
    )
    if not summary_jsons:
        raise RuntimeError(f"no summary json found in {SNAPSHOT_DIR}")

    rows = []

    for jpath in summary_jsons:
        noise_label = parse_noise_label_from_filename(jpath.name)
        noise_db = noise_label_to_db(noise_label)

        bdirs = list_valid_bundle_dirs_for_noise(noise_label)

        bundle_stats = []
        failed_bundles = []

        for bdir in bdirs:
            try:
                st = compute_llr_proxy_for_bundle(bdir)
                bundle_stats.append(st)
            except Exception as e:
                failed_bundles.append((bdir.name, str(e)))

        if bundle_stats:
            abs_mean_arr = np.array([x["llr_abs_mean"] for x in bundle_stats], dtype=np.float64)
            abs_rms_arr = np.array([x["llr_abs_rms"] for x in bundle_stats], dtype=np.float64)
            abs_med_arr = np.array([x["llr_abs_median"] for x in bundle_stats], dtype=np.float64)
            abs_p95_arr = np.array([x["llr_abs_p95"] for x in bundle_stats], dtype=np.float64)
            llr_std_arr = np.array([x["llr_std"] for x in bundle_stats], dtype=np.float64)

            row = {
                "noise_label": noise_label,
                "noise_db": noise_db,
                "used_bundle_count": len(bdirs),
                "proxy_bundle_count": len(bundle_stats),
                "proxy_fail_count": len(failed_bundles),

                "llr_abs_mean_mean": float(np.mean(abs_mean_arr)),
                "llr_abs_mean_std": float(np.std(abs_mean_arr)),
                "llr_abs_mean_median": float(median(abs_mean_arr)),
                "llr_abs_mean_min": float(np.min(abs_mean_arr)),
                "llr_abs_mean_max": float(np.max(abs_mean_arr)),

                "llr_abs_rms_mean": float(np.mean(abs_rms_arr)),
                "llr_abs_rms_std": float(np.std(abs_rms_arr)),
                "llr_abs_rms_median": float(median(abs_rms_arr)),

                "llr_abs_median_mean": float(np.mean(abs_med_arr)),
                "llr_abs_p95_mean": float(np.mean(abs_p95_arr)),
                "llr_std_mean": float(np.mean(llr_std_arr)),
                "fail_examples": failed_bundles[:3],
            }
        else:
            row = {
                "noise_label": noise_label,
                "noise_db": noise_db,
                "used_bundle_count": len(bdirs),
                "proxy_bundle_count": 0,
                "proxy_fail_count": len(failed_bundles),

                "llr_abs_mean_mean": math.nan,
                "llr_abs_mean_std": math.nan,
                "llr_abs_mean_median": math.nan,
                "llr_abs_mean_min": math.nan,
                "llr_abs_mean_max": math.nan,

                "llr_abs_rms_mean": math.nan,
                "llr_abs_rms_std": math.nan,
                "llr_abs_rms_median": math.nan,

                "llr_abs_median_mean": math.nan,
                "llr_abs_p95_mean": math.nan,
                "llr_std_mean": math.nan,
                "fail_examples": failed_bundles[:3],
            }

        rows.append(row)

    return rows


def write_proxy_csv(rows, out_csv: Path):
    header = [
        "noise_db",
        "noise_label",
        "used_bundle_count",
        "proxy_bundle_count",
        "proxy_fail_count",
        "llr_abs_mean_mean",
        "llr_abs_mean_std",
        "llr_abs_mean_median",
        "llr_abs_mean_min",
        "llr_abs_mean_max",
        "llr_abs_rms_mean",
        "llr_abs_rms_std",
        "llr_abs_rms_median",
        "llr_abs_median_mean",
        "llr_abs_p95_mean",
        "llr_std_mean",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                f"{r['noise_db']:.6f}",
                r["noise_label"],
                r["used_bundle_count"],
                r["proxy_bundle_count"],
                r["proxy_fail_count"],
                f"{r['llr_abs_mean_mean']:.12f}" if math.isfinite(r["llr_abs_mean_mean"]) else "",
                f"{r['llr_abs_mean_std']:.12f}" if math.isfinite(r["llr_abs_mean_std"]) else "",
                f"{r['llr_abs_mean_median']:.12f}" if math.isfinite(r["llr_abs_mean_median"]) else "",
                f"{r['llr_abs_mean_min']:.12f}" if math.isfinite(r["llr_abs_mean_min"]) else "",
                f"{r['llr_abs_mean_max']:.12f}" if math.isfinite(r["llr_abs_mean_max"]) else "",
                f"{r['llr_abs_rms_mean']:.12f}" if math.isfinite(r["llr_abs_rms_mean"]) else "",
                f"{r['llr_abs_rms_std']:.12f}" if math.isfinite(r["llr_abs_rms_std"]) else "",
                f"{r['llr_abs_rms_median']:.12f}" if math.isfinite(r["llr_abs_rms_median"]) else "",
                f"{r['llr_abs_median_mean']:.12f}" if math.isfinite(r["llr_abs_median_mean"]) else "",
                f"{r['llr_abs_p95_mean']:.12f}" if math.isfinite(r["llr_abs_p95_mean"]) else "",
                f"{r['llr_std_mean']:.12f}" if math.isfinite(r["llr_std_mean"]) else "",
            ])


def read_ber_summary_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "noise_db": float(row["noise_db"]),
                "noise_label": row["noise_label"],
                "group_ber_pred": float(row["group_ber_pred"]),
                "used_bundle_count": int(row["total_after_take"]) if "total_after_take" in row else None,
            })
    return rows


def join_ber_and_proxy(ber_rows, proxy_rows, x_key="llr_abs_mean_mean"):
    proxy_map = {r["noise_label"]: r for r in proxy_rows}
    out = []

    for b in ber_rows:
        p = proxy_map.get(b["noise_label"])
        if p is None:
            continue

        x = p.get(x_key, math.nan)
        out.append({
            "noise_label": b["noise_label"],
            "noise_db": b["noise_db"],
            "group_ber_pred": b["group_ber_pred"],
            "proxy_name": x_key,
            "proxy_x": x,
            "proxy_bundle_count": p["proxy_bundle_count"],
            "proxy_fail_count": p["proxy_fail_count"],
            "llr_abs_mean_mean": p["llr_abs_mean_mean"],
            "llr_abs_rms_mean": p["llr_abs_rms_mean"],
            "llr_abs_p95_mean": p["llr_abs_p95_mean"],
            "llr_std_mean": p["llr_std_mean"],
        })

    out.sort(key=lambda x: x["proxy_x"] if math.isfinite(x["proxy_x"]) else -1e18)
    return out


def write_ber_vs_proxy_csv(rows, out_csv: Path):
    header = [
        "noise_db",
        "noise_label",
        "group_ber_pred",
        "proxy_name",
        "proxy_x",
        "proxy_bundle_count",
        "proxy_fail_count",
        "llr_abs_mean_mean",
        "llr_abs_rms_mean",
        "llr_abs_p95_mean",
        "llr_std_mean",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                f"{r['noise_db']:.6f}",
                r["noise_label"],
                f"{r['group_ber_pred']:.12e}",
                r["proxy_name"],
                f"{r['proxy_x']:.12f}" if math.isfinite(r["proxy_x"]) else "",
                r["proxy_bundle_count"],
                r["proxy_fail_count"],
                f"{r['llr_abs_mean_mean']:.12f}" if math.isfinite(r["llr_abs_mean_mean"]) else "",
                f"{r['llr_abs_rms_mean']:.12f}" if math.isfinite(r["llr_abs_rms_mean"]) else "",
                f"{r['llr_abs_p95_mean']:.12f}" if math.isfinite(r["llr_abs_p95_mean"]) else "",
                f"{r['llr_std_mean']:.12f}" if math.isfinite(r["llr_std_mean"]) else "",
            ])


def plot_ber_vs_proxy(rows, out_png: Path, x_label: str):
    rows_ok = [r for r in rows if math.isfinite(r["proxy_x"])]
    if not rows_ok:
        raise RuntimeError("no finite proxy rows to plot")

    x = [r["proxy_x"] for r in rows_ok]
    y = [r["group_ber_pred"] for r in rows_ok]
    y_plot = [v if v > 0.0 else 1e-12 for v in y]

    plt.figure(figsize=(8, 5))
    plt.semilogy(x, y_plot, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Group BER")
    plt.title("SpikingRx BER vs LLR Reliability Proxy")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    proxy_rows = summarize_noise_to_proxy()

    print("\n=== Noise -> LLR Reliability Proxy Summary ===")
    print(
        f"{'noise':>8} | {'used':>6} | {'proxy_ok':>8} | {'proxy_fail':>10} | "
        f"{'abs_mean':>10} | {'abs_rms':>10} | {'p95':>10}"
    )
    print("-" * 90)

    for r in proxy_rows:
        a = r["llr_abs_mean_mean"]
        b = r["llr_abs_rms_mean"]
        c = r["llr_abs_p95_mean"]

        a_s = f"{a:.6f}" if math.isfinite(a) else "nan"
        b_s = f"{b:.6f}" if math.isfinite(b) else "nan"
        c_s = f"{c:.6f}" if math.isfinite(c) else "nan"

        print(
            f"{format_db_value(r['noise_db']):>8} | "
            f"{r['used_bundle_count']:>6d} | "
            f"{r['proxy_bundle_count']:>8d} | "
            f"{r['proxy_fail_count']:>10d} | "
            f"{a_s:>10} | "
            f"{b_s:>10} | "
            f"{c_s:>10}"
        )

        if r["proxy_fail_count"] > 0 and r["fail_examples"]:
            print(f"    fail_examples: {r['fail_examples']}")

    # historical filename kept for compatibility
    out_proxy_csv_compat = SNAPSHOT_DIR / "noise_to_snr_summary.csv"
    write_proxy_csv(proxy_rows, out_proxy_csv_compat)

    ber_summary_csv = SNAPSHOT_DIR / "noise_sweep_summary.csv"
    if not ber_summary_csv.exists():
        raise FileNotFoundError(
            f"missing BER summary csv: {ber_summary_csv}\n"
            f"run summarize_noise_sweep.py first"
        )

    ber_rows = read_ber_summary_csv(ber_summary_csv)

    # default proxy axis
    proxy_key = "llr_abs_mean_mean"
    ber_vs_proxy_rows = join_ber_and_proxy(ber_rows, proxy_rows, x_key=proxy_key)

    out_ber_vs_proxy_csv = SNAPSHOT_DIR / "ber_vs_llr_proxy.csv"
    write_ber_vs_proxy_csv(ber_vs_proxy_rows, out_ber_vs_proxy_csv)

    out_ber_vs_proxy_png = SNAPSHOT_DIR / "ber_vs_llr_proxy.png"
    plot_ber_vs_proxy(
        ber_vs_proxy_rows,
        out_ber_vs_proxy_png,
        x_label="Mean |LLR| (reliability proxy)",
    )

    print(f"\n[OK] wrote proxy csv (compat name) : {out_proxy_csv_compat}")
    print(f"[OK] wrote ber_vs_proxy csv        : {out_ber_vs_proxy_csv}")
    print(f"[OK] wrote ber_vs_proxy png        : {out_ber_vs_proxy_png}")


if __name__ == "__main__":
    main()
