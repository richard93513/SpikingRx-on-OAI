#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
import math

import matplotlib.pyplot as plt


SNAPSHOT_DIR = Path.home() / "SpikingRx-on-OAI" / "spx_records" / "snapshots"

# ------------------------------------------------------------
# valid-bundle policy
# ------------------------------------------------------------
# 1) 先依 summary item 判斷是否為有效 bundle
# 2) 再只取前 N 個有效 bundle
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


def parse_noise_label_from_filename(name: str) -> str:
    m = re.match(r"spikingrx_batch_summary_noise_power_(.+)\.json$", name)
    if not m:
        raise ValueError(f"unexpected summary filename: {name}")
    return m.group(1)


def noise_label_to_db(label: str) -> float:
    """
    支援:
      0dB
      2dB
      2.5dB
      minus2dB
      minus4.5dB
      minus4.8dB
      minus4.9dB
    """
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


def find_bundle_list(summary_obj):
    if isinstance(summary_obj, dict):
        if "bundles" in summary_obj and isinstance(summary_obj["bundles"], list):
            return summary_obj["bundles"]
        if "results" in summary_obj and isinstance(summary_obj["results"], list):
            return summary_obj["results"]
        if "items" in summary_obj and isinstance(summary_obj["items"], list):
            return summary_obj["items"]

    raise RuntimeError("cannot find bundle list in summary json")


def get_bundle_name(item: dict, fallback_idx: int) -> str:
    for k in ["bundle", "bundle_name", "bundle_dir", "name", "bdir"]:
        if k in item:
            return str(item[k])
    return f"<bundle_{fallback_idx:04d}>"


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
    """
    跟 inference 正式統計邏輯對齊：
    - status 必須 OK/ok
    - decode_script_rc 必須為 0（若該欄位存在）
    - 必須有 pred 統計欄位
    """
    status = get_status(item)
    if status not in ("ok",):
        return False

    rc = get_decode_script_rc(item)
    if rc is not None and rc != 0:
        return False

    if not has_required_pred_fields(item):
        return False

    return True


def get_bit_errors_pred(item: dict) -> int:
    if "bit_errors_pred" in item:
        return int(item["bit_errors_pred"])
    if "bit_errors" in item:
        return int(item["bit_errors"])
    if "errors" in item:
        return int(item["errors"])
    raise RuntimeError(f"missing predicted bit-errors field in item: {item.keys()}")


def get_total_bits(item: dict) -> int:
    for k in ["A", "total_bits", "n_bits", "bits"]:
        if k in item:
            return int(item[k])

    return 9480


def get_ber_pred(item: dict, bit_errors: int, total_bits: int) -> float:
    if "ber_pred" in item:
        return float(item["ber_pred"])
    if "ber" in item:
        return float(item["ber"])
    if total_bits <= 0:
        return 1.0
    return float(bit_errors) / float(total_bits)


def summarize_one_json(jpath: Path) -> dict:
    obj = json.loads(jpath.read_text())
    bundles_all = find_bundle_list(obj)

    noise_label = parse_noise_label_from_filename(jpath.name)
    noise_db = noise_label_to_db(noise_label)
    take_n = TAKE_OVERRIDE.get(noise_label, DEFAULT_TAKE_FIRST_VALID_N)

    total_raw = len(bundles_all)

    valid_items = []
    invalid_count = 0

    for i, item in enumerate(bundles_all):
        if is_valid_bundle(item):
            valid_items.append((i, item))
        else:
            invalid_count += 1

    bundles_kept = valid_items[:take_n]
    total_valid = len(valid_items)
    total_after_take = len(bundles_kept)

    total_bit_errors_pred = 0
    total_bits_pred = 0
    nonzero_bundle_count_pred = 0

    worst_bundle_name = None
    worst_bundle_ber_pred = -1.0
    first_nonzero_bundle = None
    first_valid_bundle_name = None

    for orig_idx, item in bundles_kept:
        bname = get_bundle_name(item, orig_idx)

        if first_valid_bundle_name is None:
            first_valid_bundle_name = bname

        bit_errors_pred = get_bit_errors_pred(item)
        bits = get_total_bits(item)
        ber_pred = get_ber_pred(item, bit_errors_pred, bits)

        total_bit_errors_pred += bit_errors_pred
        total_bits_pred += bits

        if bit_errors_pred > 0:
            nonzero_bundle_count_pred += 1
            if first_nonzero_bundle is None:
                first_nonzero_bundle = {
                    "name": bname,
                    "orig_index": orig_idx,
                    "bit_errors_pred": bit_errors_pred,
                    "bits": bits,
                    "ber_pred": ber_pred,
                }

        if ber_pred > worst_bundle_ber_pred:
            worst_bundle_ber_pred = ber_pred
            worst_bundle_name = bname

    group_ber_pred = (
        total_bit_errors_pred / total_bits_pred
        if total_bits_pred > 0 else math.nan
    )

    return {
        "noise_label": noise_label,
        "noise_db": noise_db,
        "take_n": take_n,
        "json_path": str(jpath),

        "total_raw": total_raw,
        "total_valid": total_valid,
        "invalid_count": invalid_count,
        "total_after_take": total_after_take,

        "nonzero_bundle_count_pred": nonzero_bundle_count_pred,
        "total_bit_errors_pred": total_bit_errors_pred,
        "total_bits_pred": total_bits_pred,
        "group_ber_pred": group_ber_pred,

        "first_valid_bundle_name": first_valid_bundle_name,
        "worst_bundle_name": worst_bundle_name,
        "worst_bundle_ber_pred": worst_bundle_ber_pred,
        "first_nonzero_bundle": first_nonzero_bundle,
    }


def print_main_table(rows):
    print("\n=== Noise Sweep Summary (Pred main result) ===")
    print(
        f"{'noise':>8} | {'take':>4} | {'raw':>6} | {'valid':>6} | {'used':>6} | "
        f"{'nz_pred':>7} | {'err_pred':>10} | {'bits_pred':>10} | {'BER_pred':>12}"
    )
    print("-" * 97)

    for r in rows:
        print(
            f"{format_db_value(r['noise_db']):>8} | "
            f"{r['take_n']:>4d} | "
            f"{r['total_raw']:>6d} | "
            f"{r['total_valid']:>6d} | "
            f"{r['total_after_take']:>6d} | "
            f"{r['nonzero_bundle_count_pred']:>7d} | "
            f"{r['total_bit_errors_pred']:>10d} | "
            f"{r['total_bits_pred']:>10d} | "
            f"{r['group_ber_pred']:>12.6e}"
        )


def print_debug(rows):
    print("\n=== Debug / Valid-bundle filtered summary ===")
    for r in rows:
        print(f"\n[noise {format_db_value(r['noise_db'])} dB]")
        print(f"  summary_json          = {r['json_path']}")
        print(f"  take_n                = {r['take_n']}")
        print(f"  total_raw             = {r['total_raw']}")
        print(f"  total_valid           = {r['total_valid']}")
        print(f"  invalid_count         = {r['invalid_count']}")
        print(f"  total_after_take      = {r['total_after_take']}")
        print(f"  first_valid_bundle    = {r['first_valid_bundle_name']}")
        print(f"  worst_bundle          = {r['worst_bundle_name']}")
        print(f"  worst_ber_pred        = {r['worst_bundle_ber_pred']:.6f}")

        if r["first_nonzero_bundle"] is None:
            print("  first_nonzero_bundle  = None")
        else:
            x = r["first_nonzero_bundle"]
            print("  first_nonzero_bundle:")
            print(f"    name               = {x['name']}")
            print(f"    orig_index         = {x['orig_index']}")
            print(f"    bit_errors_pred    = {x['bit_errors_pred']}")
            print(f"    bits               = {x['bits']}")
            print(f"    ber_pred           = {x['ber_pred']:.6f}")


def write_csv(rows, out_csv: Path):
    header = [
        "noise_db",
        "noise_label",
        "take_n",
        "total_raw",
        "total_valid",
        "invalid_count",
        "total_after_take",
        "nonzero_bundle_count_pred",
        "total_bit_errors_pred",
        "total_bits_pred",
        "group_ber_pred",
        "first_valid_bundle_name",
        "worst_bundle_name",
        "worst_bundle_ber_pred",
        "first_nonzero_bundle_name",
        "first_nonzero_bundle_orig_index",
        "first_nonzero_bundle_bit_errors_pred",
        "first_nonzero_bundle_bits",
        "first_nonzero_bundle_ber_pred",
    ]

    lines = [",".join(header)]

    for r in rows:
        first = r["first_nonzero_bundle"] or {}

        vals = [
            f"{r['noise_db']:.6f}",
            r["noise_label"],
            str(r["take_n"]),
            str(r["total_raw"]),
            str(r["total_valid"]),
            str(r["invalid_count"]),
            str(r["total_after_take"]),
            str(r["nonzero_bundle_count_pred"]),
            str(r["total_bit_errors_pred"]),
            str(r["total_bits_pred"]),
            f"{r['group_ber_pred']:.12e}",
            str(r["first_valid_bundle_name"]),
            str(r["worst_bundle_name"]),
            f"{r['worst_bundle_ber_pred']:.12e}",
            str(first.get("name", "")),
            str(first.get("orig_index", "")),
            str(first.get("bit_errors_pred", "")),
            str(first.get("bits", "")),
            (f"{first.get('ber_pred', float('nan')):.12e}" if first else ""),
        ]
        lines.append(",".join(vals))

    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_ber_curve(rows, out_png: Path):
    x = [r["noise_db"] for r in rows]
    y = [r["group_ber_pred"] for r in rows]

    # 全域圖保留 semilogy
    y_plot = [v if (isinstance(v, float) and v > 0.0) else 1e-12 for v in y]

    plt.figure(figsize=(8, 5))
    plt.semilogy(x, y_plot, marker="o")
    plt.xlabel("noise_power_dB")
    plt.ylabel("Group BER")
    plt.title("SpikingRx Noise Sweep BER")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ber_curve_zoom_left(rows, out_png: Path, x_min: float = -5.0, x_max: float = -4.0):
    """
    左肩局部圖：
    - 只看 x in [x_min, x_max]
    - 用線性 y 軸，比較容易看清 0 / 3e-4 / 1e-3 / 7e-3 / 3e-2
    """
    rows_zoom = [r for r in rows if x_min <= r["noise_db"] <= x_max]
    if not rows_zoom:
        return

    x = [r["noise_db"] for r in rows_zoom]
    y = [r["group_ber_pred"] for r in rows_zoom]

    ymax = max(y) if y else 0.0
    if ymax <= 0.0:
        ymax = 1e-3

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("noise_power_dB")
    plt.ylabel("Group BER")
    plt.title("SpikingRx Noise Sweep BER (Zoom: Left Shoulder)")
    plt.xlim(x_min - 0.02, x_max + 0.02)
    plt.ylim(-0.001, ymax * 1.10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if not SNAPSHOT_DIR.exists():
        raise SystemExit(f"[ERROR] snapshot dir not found: {SNAPSHOT_DIR}")

    json_list = sorted(
        SNAPSHOT_DIR.glob("spikingrx_batch_summary_noise_power_*.json"),
        key=lambda p: noise_label_to_db(parse_noise_label_from_filename(p.name))
    )

    if not json_list:
        raise SystemExit(f"[ERROR] no summary json found in: {SNAPSHOT_DIR}")

    rows = []
    for jpath in json_list:
        rows.append(summarize_one_json(jpath))

    print_main_table(rows)
    print_debug(rows)

    out_csv = SNAPSHOT_DIR / "noise_sweep_summary.csv"
    write_csv(rows, out_csv)

    out_png = SNAPSHOT_DIR / "noise_sweep_ber_curve.png"
    plot_ber_curve(rows, out_png)

    out_png_zoom = SNAPSHOT_DIR / "noise_sweep_ber_curve_zoom_left.png"
    plot_ber_curve_zoom_left(rows, out_png_zoom, x_min=-5.0, x_max=-4.0)

    print(f"\n[OK] wrote csv      : {out_csv}")
    print(f"[OK] wrote plot     : {out_png}")
    print(f"[OK] wrote zoom plot: {out_png_zoom}")


if __name__ == "__main__":
    main()
