#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import math
import re

import matplotlib.pyplot as plt


REPO_ROOT = Path.home() / "SpikingRx-on-OAI"
SNAPSHOT_DIR = REPO_ROOT / "spx_records" / "snapshots_snr"

RE_SNR_LINE = re.compile(
    r"noise_power_db=(?P<noise>[-+]?\d+(?:\.\d+)?)\s+"
    r"snr_top10_db=(?P<snr>[-+]?\d+(?:\.\d+)?)\s+"
    r"used_windows=(?P<windows>\d+)"
)


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
    if abs(x * 10 - round(x * 10)) < 1e-9:
        return f"{x:.1f}"
    return f"{x:.2f}"


def read_noise_sweep_summary_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "noise_db": float(row["noise_db"]),
                "noise_label": row["noise_label"],
                "group_ber_pred": float(row["group_ber_pred"]),
                "total_after_take": int(row["total_after_take"]) if row.get("total_after_take") else None,
                "total_valid": int(row["total_valid"]) if row.get("total_valid") else None,
            })
    return rows


def read_snr_summary_txt(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing SNR summary: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"empty SNR summary: {path}")

    # 正常情況只有一行；若之後不小心 append 多行，就取最後一行
    last = lines[-1]
    m = RE_SNR_LINE.search(last)
    if not m:
        raise RuntimeError(f"cannot parse SNR summary line in {path}: {last}")

    return {
        "noise_power_db": float(m.group("noise")),
        "snr_top10_db": float(m.group("snr")),
        "used_windows": int(m.group("windows")),
        "raw_line": last,
    }


def summarize_noise_to_snr():
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

        bundle_root = SNAPSHOT_DIR / f"bundle_noise_power_{noise_label}"
        snr_txt = bundle_root / "snr_top10_run_summary.txt"

        snr_info = read_snr_summary_txt(snr_txt)

        rows.append({
            "noise_label": noise_label,
            "noise_db_from_name": noise_db,
            "noise_power_db_from_file": snr_info["noise_power_db"],
            "snr_top10_db": snr_info["snr_top10_db"],
            "used_windows": snr_info["used_windows"],
            "snr_summary_path": str(snr_txt),
            "snr_summary_line": snr_info["raw_line"],
        })

    return rows


def write_noise_to_snr_csv(rows, out_csv: Path):
    header = [
        "noise_db_from_name",
        "noise_label",
        "noise_power_db_from_file",
        "snr_top10_db",
        "used_windows",
        "snr_summary_path",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                f"{r['noise_db_from_name']:.6f}",
                r["noise_label"],
                f"{r['noise_power_db_from_file']:.6f}",
                f"{r['snr_top10_db']:.12f}",
                r["used_windows"],
                r["snr_summary_path"],
            ])


def join_ber_and_snr(ber_rows, snr_rows):
    snr_map = {r["noise_label"]: r for r in snr_rows}
    out = []

    for b in ber_rows:
        s = snr_map.get(b["noise_label"])
        if s is None:
            continue

        out.append({
            "noise_label": b["noise_label"],
            "noise_db": b["noise_db"],
            "snr_db": s["snr_top10_db"],
            "group_ber_pred": b["group_ber_pred"],
            "used_windows": s["used_windows"],
            "total_after_take": b["total_after_take"],
            "total_valid": b["total_valid"],
        })

    out.sort(key=lambda x: x["snr_db"])
    return out


def write_ber_vs_snr_csv(rows, out_csv: Path):
    header = [
        "snr_db",
        "noise_db",
        "noise_label",
        "group_ber_pred",
        "used_windows",
        "total_after_take",
        "total_valid",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                f"{r['snr_db']:.12f}",
                f"{r['noise_db']:.6f}",
                r["noise_label"],
                f"{r['group_ber_pred']:.12e}",
                r["used_windows"] if r["used_windows"] is not None else "",
                r["total_after_take"] if r["total_after_take"] is not None else "",
                r["total_valid"] if r["total_valid"] is not None else "",
            ])


def plot_ber_vs_snr(rows, out_png: Path):
    rows_ok = [r for r in rows if math.isfinite(r["snr_db"])]
    if not rows_ok:
        raise RuntimeError("no finite SNR rows to plot")

    x = [r["snr_db"] for r in rows_ok]
    y = [r["group_ber_pred"] for r in rows_ok]
    y_plot = [v if v > 0.0 else 1e-12 for v in y]

    plt.figure(figsize=(8, 5))
    plt.semilogy(x, y_plot, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Group BER")
    plt.title("SpikingRx BER vs SNR")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    snr_rows = summarize_noise_to_snr()

    print("\n=== Noise -> SNR Summary ===")
    print(f"{'noise':>8} | {'snr_db':>12} | {'windows':>7}")
    print("-" * 36)
    for r in snr_rows:
        print(
            f"{format_db_value(r['noise_db_from_name']):>8} | "
            f"{r['snr_top10_db']:>12.6f} | "
            f"{r['used_windows']:>7d}"
        )

    out_noise_to_snr_csv = SNAPSHOT_DIR / "noise_to_snr_summary.csv"
    write_noise_to_snr_csv(snr_rows, out_noise_to_snr_csv)

    ber_summary_csv = SNAPSHOT_DIR / "noise_sweep_summary.csv"
    if not ber_summary_csv.exists():
        raise FileNotFoundError(
            f"missing BER summary csv: {ber_summary_csv}\n"
            f"run summarize_noise_sweep.py first"
        )

    ber_rows = read_noise_sweep_summary_csv(ber_summary_csv)
    ber_vs_snr_rows = join_ber_and_snr(ber_rows, snr_rows)

    out_ber_vs_snr_csv = SNAPSHOT_DIR / "ber_vs_snr.csv"
    write_ber_vs_snr_csv(ber_vs_snr_rows, out_ber_vs_snr_csv)

    out_ber_vs_snr_png = SNAPSHOT_DIR / "ber_vs_snr.png"
    plot_ber_vs_snr(ber_vs_snr_rows, out_ber_vs_snr_png)

    print(f"\n[OK] wrote noise_to_snr csv : {out_noise_to_snr_csv}")
    print(f"[OK] wrote ber_vs_snr csv   : {out_ber_vs_snr_csv}")
    print(f"[OK] wrote ber_vs_snr png   : {out_ber_vs_snr_png}")


if __name__ == "__main__":
    main()
