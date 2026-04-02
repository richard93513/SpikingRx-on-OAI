#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
NC = 1600


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


def parse_kv_file(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue

        if "=" in s:
            k, v = s.split("=", 1)
        else:
            p = s.split()
            if len(p) != 2:
                continue
            k, v = p

        k = k.strip()
        v = v.strip()

        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v, 0)
        except Exception:
            out[k] = v

    return out


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


def load_fullgrid_used(path: Path):
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size < 8:
        raise RuntimeError(f"file too small: {path}")

    header = raw[:8].view(np.uint16)
    frame = int(header[0])
    slot = int(header[1])
    start_symbol = int(header[2])
    n_sym = int(header[3])
    first_sc = int(header[4])
    used_sc = int(header[5])
    number_symbols = int(header[6])
    G = int(header[7])

    iq = raw[8:]
    if iq.size % 2 != 0:
        raise RuntimeError(f"odd IQ length in {path}")

    total_re = iq.size // 2
    if n_sym <= 0 or total_re % n_sym != 0:
        raise RuntimeError(f"bad fullgrid dimensions in {path}")

    n_sc_full = total_re // n_sym

    cpx = iq.reshape(-1, 2).astype(np.float32)
    cpx = cpx[:, 0] + 1j * cpx[:, 1]
    grid_full = cpx.reshape(n_sym, n_sc_full)

    if used_sc <= 0 or used_sc > n_sc_full:
        raise RuntimeError(f"bad used_sc={used_sc} in {path}")

    first_carrier_offset = n_sc_full - (used_sc // 2)
    idx = (np.arange(used_sc, dtype=np.int64) + first_carrier_offset) % n_sc_full
    grid_used = grid_full[:, idx].astype(np.complex64)

    meta = {
        "frame": frame,
        "slot": slot,
        "start_symbol": start_symbol,
        "n_sym": n_sym,
        "first_sc": first_sc,
        "used_sc": used_sc,
        "number_symbols": number_symbols,
        "G": G,
        "n_sc_full": n_sc_full,
        "first_carrier_offset": first_carrier_offset,
    }
    return grid_used, meta


def nr_prbs(c_init: int, length: int) -> np.ndarray:
    x1 = np.zeros(NC + length + 31, dtype=np.uint8)
    x2 = np.zeros(NC + length + 31, dtype=np.uint8)

    x1[0] = 1

    for i in range(31):
        x2[i] = (c_init >> i) & 1

    for n in range(NC + length):
        x1[n + 31] = x1[n + 3] ^ x1[n]
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]

    c = x1[NC:NC + length] ^ x2[NC:NC + length]
    return c.astype(np.uint8)


def pdsch_dmrs_cinit(slot: int, symbol: int, nid: int, nscid: int) -> int:
    c_init = ((((14 * slot + symbol + 1) * (2 * nid + 1)) << 17) + (2 * nid + nscid)) & 0x7FFFFFFF
    return int(c_init)


def gen_pdsch_dmrs_qpsk(slot: int, symbol: int, nid: int, nscid: int, n_re: int) -> np.ndarray:
    c_init = pdsch_dmrs_cinit(slot, symbol, nid, nscid)
    c = nr_prbs(c_init, 2 * n_re)

    re_part = (1.0 - 2.0 * c[0::2].astype(np.float32)) / np.sqrt(2.0)
    im_part = (1.0 - 2.0 * c[1::2].astype(np.float32)) / np.sqrt(2.0)
    return (re_part + 1j * im_part).astype(np.complex64)


def get_dmrs_symbol_list(dlDmrsSymbPos: int, n_sym: int):
    return [s for s in range(n_sym) if ((dlDmrsSymbPos >> s) & 0x1) != 0]


def get_dmrs_re_pattern(dmrsConfigType: int, n_dmrs_cdm_groups: int):
    if dmrsConfigType == 0 and n_dmrs_cdm_groups == 1:
        return [0, 2, 4, 6, 8, 10]
    elif dmrsConfigType == 1 and n_dmrs_cdm_groups == 1:
        return [0, 1, 6, 7]
    elif dmrsConfigType == 1 and n_dmrs_cdm_groups == 2:
        return [0, 1, 2, 3, 6, 7, 8, 9]
    else:
        raise RuntimeError(
            f"unsupported DMRS config: dmrsConfigType={dmrsConfigType}, "
            f"n_dmrs_cdm_groups={n_dmrs_cdm_groups}"
        )


def estimate_bundle_dmrs_snr_db(bdir: Path) -> float:
    pdsch_cfg = parse_kv_file(bdir / "pdsch_cfg.txt")
    fullgrid, meta = load_fullgrid_used(bdir / "fullgrid.bin")

    slot = int(pdsch_cfg["slot"]) if "slot" in pdsch_cfg else int(meta["slot"])
    n_sym = int(meta["n_sym"])
    used_sc = int(meta["used_sc"])

    dlDmrsSymbPos = int(pdsch_cfg["dlDmrsSymbPos"])
    dmrsConfigType = int(pdsch_cfg["dmrsConfigType"])
    n_dmrs_cdm_groups = int(pdsch_cfg["n_dmrs_cdm_groups"])

    start_symbol = int(pdsch_cfg["start_symbol"])
    number_symbols = int(pdsch_cfg["number_symbols"])
    BWPStart = int(pdsch_cfg.get("BWPStart", 0))
    start_rb = int(pdsch_cfg.get("start_rb", 0))
    number_rbs = int(pdsch_cfg["number_rbs"])

    dlDmrsScramblingId = int(pdsch_cfg["dlDmrsScramblingId"])
    nscid = int(pdsch_cfg["nscid"])

    dmrs_symbols_all = get_dmrs_symbol_list(dlDmrsSymbPos, n_sym)
    dmrs_symbols = [s for s in dmrs_symbols_all if start_symbol <= s < (start_symbol + number_symbols)]
    dmrs_re_pattern = get_dmrs_re_pattern(dmrsConfigType, n_dmrs_cdm_groups)

    local_total_rbs = used_sc // 12
    local_start_rb = BWPStart + start_rb
    local_number_rbs = number_rbs

    if local_start_rb + local_number_rbs > local_total_rbs:
        raise RuntimeError(
            f"RB range exceeds used grid in {bdir}: "
            f"start={local_start_rb}, n={local_number_rbs}, total={local_total_rbs}"
        )

    signal_energy = 0.0
    noise_energy = 0.0
    n_dmrs_used = 0

    for s in dmrs_symbols:
        n_re_symbol = local_number_rbs * len(dmrs_re_pattern)
        x_symbol = gen_pdsch_dmrs_qpsk(
            slot=slot,
            symbol=s,
            nid=dlDmrsScramblingId,
            nscid=nscid,
            n_re=n_re_symbol,
        )

        ptr = 0
        for rb in range(local_start_rb, local_start_rb + local_number_rbs):
            rb_base = rb * 12
            x_rb = x_symbol[ptr:ptr + len(dmrs_re_pattern)]
            ptr += len(dmrs_re_pattern)

            y_rb = np.array(
                [fullgrid[s, rb_base + re] for re in dmrs_re_pattern],
                dtype=np.complex64
            )

            h_rb = np.mean(y_rb / x_rb)
            y_hat_rb = h_rb * x_rb
            err_rb = y_rb - y_hat_rb

            signal_energy += float(np.sum(np.abs(y_hat_rb) ** 2))
            noise_energy += float(np.sum(np.abs(err_rb) ** 2))
            n_dmrs_used += int(len(dmrs_re_pattern))

    if n_dmrs_used <= 0:
        raise RuntimeError(f"no DMRS RE used in {bdir}")

    if noise_energy <= 0.0:
        return float("inf")

    snr_lin = signal_energy / max(noise_energy, EPS)
    snr_db = 10.0 * math.log10(max(snr_lin, EPS))
    return snr_db


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

        bdirs = list_valid_bundle_dirs_for_noise(noise_label)
        snr_list = []
        failed_bundles = []

        for bdir in bdirs:
            try:
                snr_db = estimate_bundle_dmrs_snr_db(bdir)
                if math.isfinite(snr_db):
                    snr_list.append(snr_db)
            except Exception as e:
                failed_bundles.append((bdir.name, str(e)))

        if snr_list:
            mean_snr = float(np.mean(snr_list))
            std_snr = float(np.std(snr_list))
            med_snr = float(median(snr_list))
            min_snr = float(np.min(snr_list))
            max_snr = float(np.max(snr_list))
        else:
            mean_snr = math.nan
            std_snr = math.nan
            med_snr = math.nan
            min_snr = math.nan
            max_snr = math.nan

        rows.append({
            "noise_label": noise_label,
            "noise_db": noise_db,
            "used_bundle_count": len(bdirs),
            "snr_bundle_count": len(snr_list),
            "snr_fail_count": len(failed_bundles),
            "snr_mean_db": mean_snr,
            "snr_median_db": med_snr,
            "snr_std_db": std_snr,
            "snr_min_db": min_snr,
            "snr_max_db": max_snr,
            "fail_examples": failed_bundles[:3],
        })

    return rows


def write_noise_to_snr_csv(rows, out_csv: Path):
    header = [
        "noise_db",
        "noise_label",
        "used_bundle_count",
        "snr_bundle_count",
        "snr_fail_count",
        "snr_mean_db",
        "snr_median_db",
        "snr_std_db",
        "snr_min_db",
        "snr_max_db",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                f"{r['noise_db']:.6f}",
                r["noise_label"],
                r["used_bundle_count"],
                r["snr_bundle_count"],
                r["snr_fail_count"],
                f"{r['snr_mean_db']:.12f}" if math.isfinite(r["snr_mean_db"]) else "",
                f"{r['snr_median_db']:.12f}" if math.isfinite(r["snr_median_db"]) else "",
                f"{r['snr_std_db']:.12f}" if math.isfinite(r["snr_std_db"]) else "",
                f"{r['snr_min_db']:.12f}" if math.isfinite(r["snr_min_db"]) else "",
                f"{r['snr_max_db']:.12f}" if math.isfinite(r["snr_max_db"]) else "",
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
            "group_ber_pred": b["group_ber_pred"],
            "snr_mean_db": s["snr_mean_db"],
            "snr_median_db": s["snr_median_db"],
            "snr_std_db": s["snr_std_db"],
            "snr_bundle_count": s["snr_bundle_count"],
        })

    out.sort(key=lambda x: x["snr_mean_db"] if math.isfinite(x["snr_mean_db"]) else -1e9)
    return out


def write_ber_vs_snr_csv(rows, out_csv: Path):
    header = [
        "noise_db",
        "noise_label",
        "group_ber_pred",
        "snr_mean_db",
        "snr_median_db",
        "snr_std_db",
        "snr_bundle_count",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                f"{r['noise_db']:.6f}",
                r["noise_label"],
                f"{r['group_ber_pred']:.12e}",
                f"{r['snr_mean_db']:.12f}" if math.isfinite(r["snr_mean_db"]) else "",
                f"{r['snr_median_db']:.12f}" if math.isfinite(r["snr_median_db"]) else "",
                f"{r['snr_std_db']:.12f}" if math.isfinite(r["snr_std_db"]) else "",
                r["snr_bundle_count"],
            ])


def plot_ber_vs_snr(rows, out_png: Path):
    rows_ok = [r for r in rows if math.isfinite(r["snr_mean_db"])]
    if not rows_ok:
        raise RuntimeError("no finite SNR rows to plot")

    x = [r["snr_mean_db"] for r in rows_ok]
    y = [r["group_ber_pred"] for r in rows_ok]
    y_plot = [v if v > 0.0 else 1e-12 for v in y]

    plt.figure(figsize=(8, 5))
    plt.semilogy(x, y_plot, marker="o")
    plt.xlabel("DMRS-based Channel SNR (dB)")
    plt.ylabel("Group BER")
    plt.title("SpikingRx BER vs DMRS-based Channel SNR")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    noise_to_snr_rows = summarize_noise_to_snr()

    print("\n=== Noise -> DMRS-based Channel SNR Summary ===")
    print(
        f"{'noise':>8} | {'used':>6} | {'snr_ok':>6} | {'snr_fail':>8} | "
        f"{'mean_snr':>10} | {'std':>8} | {'median':>10}"
    )
    print("-" * 82)

    for r in noise_to_snr_rows:
        mean_snr = r["snr_mean_db"]
        std_snr = r["snr_std_db"]
        med_snr = r["snr_median_db"]

        mean_s = f"{mean_snr:.3f}" if math.isfinite(mean_snr) else "nan"
        std_s = f"{std_snr:.3f}" if math.isfinite(std_snr) else "nan"
        med_s = f"{med_snr:.3f}" if math.isfinite(med_snr) else "nan"

        print(
            f"{format_db_value(r['noise_db']):>8} | "
            f"{r['used_bundle_count']:>6d} | "
            f"{r['snr_bundle_count']:>6d} | "
            f"{r['snr_fail_count']:>8d} | "
            f"{mean_s:>10} | "
            f"{std_s:>8} | "
            f"{med_s:>10}"
        )

        if r["snr_fail_count"] > 0 and r["fail_examples"]:
            print(f"    fail_examples: {r['fail_examples']}")

    out_noise_to_snr_csv = SNAPSHOT_DIR / "noise_to_snr_summary.csv"
    write_noise_to_snr_csv(noise_to_snr_rows, out_noise_to_snr_csv)

    ber_summary_csv = SNAPSHOT_DIR / "noise_sweep_summary.csv"
    if not ber_summary_csv.exists():
        raise FileNotFoundError(
            f"missing BER summary csv: {ber_summary_csv}\n"
            f"run summarize_noise_sweep.py first"
        )

    ber_rows = read_ber_summary_csv(ber_summary_csv)
    ber_vs_snr_rows = join_ber_and_snr(ber_rows, noise_to_snr_rows)

    out_ber_vs_snr_csv = SNAPSHOT_DIR / "ber_vs_snr.csv"
    write_ber_vs_snr_csv(ber_vs_snr_rows, out_ber_vs_snr_csv)

    out_ber_vs_snr_png = SNAPSHOT_DIR / "ber_vs_snr.png"
    plot_ber_vs_snr(ber_vs_snr_rows, out_ber_vs_snr_png)

    print(f"\n[OK] wrote noise->snr csv : {out_noise_to_snr_csv}")
    print(f"[OK] wrote ber_vs_snr csv: {out_ber_vs_snr_csv}")
    print(f"[OK] wrote ber_vs_snr png: {out_ber_vs_snr_png}")


if __name__ == "__main__":
    main()
