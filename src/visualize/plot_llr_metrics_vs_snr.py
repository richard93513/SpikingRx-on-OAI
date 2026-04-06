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
import matplotlib.ticker as mticker
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


def read_spikingrx_summary_rows(summary_json: Path) -> List[dict]:
    obj = json.loads(summary_json.read_text())
    return find_bundle_list(obj)


def list_valid_bundle_names_for_noise(snapshot_dir: Path, noise_label: str) -> List[str]:
    summary_json = snapshot_dir / f"spikingrx_batch_summary_noise_power_{noise_label}.json"
    rows = read_spikingrx_summary_rows(summary_json)
    take_n = TAKE_OVERRIDE.get(noise_label, DEFAULT_TAKE_FIRST_VALID_N)

    valid_names: List[str] = []
    for item in rows:
        if is_valid_bundle(item):
            valid_names.append(get_bundle_name(item))
            if len(valid_names) >= take_n:
                break
    return valid_names


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


def summarize_one_noise(snapshot_dir: Path, noise_label: str, max_bundles: int | None = None) -> dict:
    bnames = list_valid_bundle_names_for_noise(snapshot_dir, noise_label)
    if max_bundles is not None:
        bnames = bnames[:max_bundles]

    bundle_root = snapshot_dir / f"bundle_noise_power_{noise_label}"

    metrics_list = []
    failed = []

    for bname in bnames:
        bdir = bundle_root / bname
        if not bdir.is_dir():
            failed.append({"bundle": bname, "reason": "bundle_dir_missing"})
            continue

        try:
            spx, oai = load_llr_pair(bdir)
            m = calc_pair_metrics(spx, oai)
            m["bundle_name"] = bname
            metrics_list.append(m)
        except Exception as e:
            failed.append({"bundle": bname, "reason": repr(e)})

    def arr_of(key: str) -> np.ndarray:
        return np.array([x[key] for x in metrics_list], dtype=np.float64)

    if metrics_list:
        mean_abs_diff_arr = arr_of("mean_abs_diff")
        corr_arr = arr_of("corr")
        sign_arr = arr_of("sign_agreement")
        rmse_arr = arr_of("rmse")
        max_abs_arr = arr_of("max_abs_diff")

        row = {
            "noise_label": noise_label,
            "noise_db": noise_label_to_db(noise_label),
            "selected_bundle_count": len(bnames),
            "ok_bundle_count": len(metrics_list),
            "fail_bundle_count": len(failed),

            "mean_abs_diff_mean": float(np.mean(mean_abs_diff_arr)),
            "mean_abs_diff_std": float(np.std(mean_abs_diff_arr)),
            "mean_abs_diff_min": float(np.min(mean_abs_diff_arr)),
            "mean_abs_diff_max": float(np.max(mean_abs_diff_arr)),

            "corr_mean": float(np.mean(corr_arr)),
            "corr_std": float(np.std(corr_arr)),
            "corr_min": float(np.min(corr_arr)),
            "corr_max": float(np.max(corr_arr)),

            "sign_agreement_mean": float(np.mean(sign_arr)),
            "sign_agreement_std": float(np.std(sign_arr)),
            "sign_agreement_min": float(np.min(sign_arr)),
            "sign_agreement_max": float(np.max(sign_arr)),

            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),

            "max_abs_diff_mean": float(np.mean(max_abs_arr)),
            "max_abs_diff_std": float(np.std(max_abs_arr)),
            "max_abs_diff_max": float(np.max(max_abs_arr)),

            "fail_examples": failed[:5],
        }
    else:
        row = {
            "noise_label": noise_label,
            "noise_db": noise_label_to_db(noise_label),
            "selected_bundle_count": len(bnames),
            "ok_bundle_count": 0,
            "fail_bundle_count": len(failed),

            "mean_abs_diff_mean": math.nan,
            "mean_abs_diff_std": math.nan,
            "mean_abs_diff_min": math.nan,
            "mean_abs_diff_max": math.nan,

            "corr_mean": math.nan,
            "corr_std": math.nan,
            "corr_min": math.nan,
            "corr_max": math.nan,

            "sign_agreement_mean": math.nan,
            "sign_agreement_std": math.nan,
            "sign_agreement_min": math.nan,
            "sign_agreement_max": math.nan,

            "rmse_mean": math.nan,
            "rmse_std": math.nan,

            "max_abs_diff_mean": math.nan,
            "max_abs_diff_std": math.nan,
            "max_abs_diff_max": math.nan,

            "fail_examples": failed[:5],
        }

    return row


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "noise_db",
        "noise_label",
        "snr_top10_db",
        "selected_bundle_count",
        "ok_bundle_count",
        "fail_bundle_count",

        "mean_abs_diff_mean",
        "mean_abs_diff_std",
        "mean_abs_diff_min",
        "mean_abs_diff_max",

        "corr_mean",
        "corr_std",
        "corr_min",
        "corr_max",

        "sign_agreement_mean",
        "sign_agreement_std",
        "sign_agreement_min",
        "sign_agreement_max",

        "rmse_mean",
        "rmse_std",

        "max_abs_diff_mean",
        "max_abs_diff_std",
        "max_abs_diff_max",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def apply_plain_decimal_yaxis(ax, decimals: int = 6) -> None:
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{decimals}f"))


def plot_metric_curve(
    rows: List[dict],
    y_key: str,
    title: str,
    ylabel: str,
    out_png: Path,
    y_limits: Tuple[float, float] | None = None,
    plain_decimal: bool = False,
    decimals: int = 6,
) -> None:
    rows_ok = [r for r in rows if math.isfinite(r["snr_top10_db"]) and math.isfinite(r[y_key])]
    if not rows_ok:
        return

    x = [r["snr_top10_db"] for r in rows_ok]
    y = [r[y_key] for r in rows_ok]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if plain_decimal:
        apply_plain_decimal_yaxis(ax, decimals=decimals)

    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_metrics(rows: List[dict], out_png: Path) -> None:
    rows_ok = [r for r in rows if math.isfinite(r["snr_top10_db"])]
    if not rows_ok:
        return

    x = np.array([r["snr_top10_db"] for r in rows_ok], dtype=np.float64)
    mad = np.array([r["mean_abs_diff_mean"] for r in rows_ok], dtype=np.float64)
    corr = np.array([r["corr_mean"] for r in rows_ok], dtype=np.float64)
    sign = np.array([r["sign_agreement_mean"] for r in rows_ok], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    axes[0].plot(x, mad, marker="o")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Mean |LLR diff|")
    axes[0].set_title("Mean absolute LLR difference vs SNR")
    apply_plain_decimal_yaxis(axes[0], decimals=5)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(x, corr, marker="o")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("LLR correlation vs SNR")
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_yticks(np.linspace(-1.0, 1.0, 9))
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].plot(x, sign, marker="o")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Sign agreement")
    axes[2].set_title("LLR sign agreement vs SNR")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_yticks(np.linspace(0.0, 1.0, 6))
    axes[2].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Summarize full-SNR-sweep LLR metrics: mean_abs_diff / corr / sign_agreement vs SNR"
    )
    ap.add_argument("--snapshot_dir", type=str, default=str(DEFAULT_SNAPSHOT_DIR))
    ap.add_argument(
        "--max_bundles_per_noise",
        type=int,
        default=None,
        help="limit bundles per noise point for faster debug; default uses all selected bundles",
    )
    args = ap.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not snapshot_dir.exists():
        raise SystemExit(f"[ERROR] snapshot dir not found: {snapshot_dir}")

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

    rows = []

    print("\n=== LLR metrics vs SNR summary ===")
    print(
        f"{'noise':>10} | {'snr_db':>10} | {'ok':>4} | {'fail':>4} | "
        f"{'mad':>10} | {'corr':>10} | {'sign_agree':>10}"
    )
    print("-" * 88)

    for jpath in summary_jsons:
        noise_label = parse_noise_label_from_summary_filename(jpath.name)

        row = summarize_one_noise(
            snapshot_dir=snapshot_dir,
            noise_label=noise_label,
            max_bundles=args.max_bundles_per_noise,
        )

        snr_row = noise_to_snr_map.get(noise_label)
        row["snr_top10_db"] = float(snr_row["snr_top10_db"]) if snr_row else math.nan

        rows.append(row)

        snr_s = "nan" if not math.isfinite(row["snr_top10_db"]) else f"{row['snr_top10_db']:.6f}"
        mad_s = "nan" if not math.isfinite(row["mean_abs_diff_mean"]) else f"{row['mean_abs_diff_mean']:.6f}"
        corr_s = "nan" if not math.isfinite(row["corr_mean"]) else f"{row['corr_mean']:.6f}"
        sign_s = "nan" if not math.isfinite(row["sign_agreement_mean"]) else f"{row['sign_agreement_mean']:.6f}"

        print(
            f"{noise_label:>10} | "
            f"{snr_s:>10} | "
            f"{row['ok_bundle_count']:>4d} | "
            f"{row['fail_bundle_count']:>4d} | "
            f"{mad_s:>10} | "
            f"{corr_s:>10} | "
            f"{sign_s:>10}"
        )

        if row["fail_bundle_count"] > 0 and row["fail_examples"]:
            print(f"    fail_examples: {row['fail_examples']}")

    rows.sort(key=lambda r: (r["snr_top10_db"] if math.isfinite(r["snr_top10_db"]) else -1e18))

    out_dir = snapshot_dir / "llr_metrics_vs_snr"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "llr_metrics_vs_snr_summary.csv"
    write_summary_csv(summary_csv, rows)

    mad_png = out_dir / "mean_abs_diff_vs_snr.png"
    plot_metric_curve(
        rows=rows,
        y_key="mean_abs_diff_mean",
        title="Mean absolute LLR difference vs SNR",
        ylabel="Mean |SpikingRx LLR - OAI LLR|",
        out_png=mad_png,
        y_limits=None,
        plain_decimal=True,
        decimals=5,
    )

    corr_png = out_dir / "corr_vs_snr.png"
    plot_metric_curve(
        rows=rows,
        y_key="corr_mean",
        title="LLR correlation vs SNR",
        ylabel="Correlation coefficient",
        out_png=corr_png,
        y_limits=(-1.0, 1.0),
        plain_decimal=False,
    )

    sign_png = out_dir / "sign_agreement_vs_snr.png"
    plot_metric_curve(
        rows=rows,
        y_key="sign_agreement_mean",
        title="LLR sign agreement vs SNR",
        ylabel="Sign agreement",
        out_png=sign_png,
        y_limits=(0.0, 1.0),
        plain_decimal=False,
    )

    combined_png = out_dir / "llr_metrics_vs_snr_combined.png"
    plot_combined_metrics(rows, combined_png)

    print("\n[OK] wrote:")
    print(f"  {summary_csv}")
    print(f"  {mad_png}")
    print(f"  {corr_png}")
    print(f"  {sign_png}")
    print(f"  {combined_png}")


if __name__ == "__main__":
    main()
