# src/train/train_spikingrx_oai.py
# -*- coding: utf-8 -*-

import os
import sys
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
DATA_DIR = os.path.join(SRC_DIR, "data")

for p in [SRC_DIR, DATA_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from data.dataset_oai_bundle import OAI_Bundle_Dataset
from models.spikingrx_model import SpikingRxModel


# -------------------------------------------------
# seed
# -------------------------------------------------
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# plot
# -------------------------------------------------
def plot_curve(epoch, train_list, val_list, title, ylabel, path):

    if len(epoch) == 0:
        return

    plt.figure()
    plt.plot(epoch, train_list, marker="o", label="train")
    plt.plot(epoch, val_list, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# -------------------------------------------------
# metrics
# -------------------------------------------------
def compute_metrics(pred, target):

    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    sign = ((pred >= 0) == (target >= 0)).float().mean()

    return {
        "mse": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "sign": float(sign.detach().cpu())
    }


# -------------------------------------------------
# train epoch
# -------------------------------------------------
def train_epoch(model, loader, optimizer, device_conv):

    model.train()

    sum_norm_mse = 0
    sum_raw_mse = 0
    sum_raw_sign = 0
    n = 0

    pbar = tqdm(
    loader,
    ncols=110,
    leave=False,
    dynamic_ncols=True,
)

    for x, y, cfg, bdir in pbar:

        x = x.to(device_conv)

        # ----------------------
        # normalize target
        # ----------------------
        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        y_norm = (y - y_mean) / (y_std + 1e-6)

        # ----------------------
        # forward
        # ----------------------
        pred_norm, aux = model(x)

        loss = torch.mean((pred_norm - y_norm) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ----------------------
        # denormalize prediction
        # ----------------------
        pred_raw = pred_norm * y_std + y_mean

        norm_mse = torch.mean((pred_norm - y_norm) ** 2)

        raw_metrics = compute_metrics(pred_raw, y)

        sum_norm_mse += float(norm_mse)
        sum_raw_mse += raw_metrics["mse"]
        sum_raw_sign += raw_metrics["sign"]

        n += 1

        pbar.set_postfix_str(
        f"norm_mse={float(norm_mse):.3f} raw_sign={raw_metrics['sign']:.3f}"
        )

    return {
        "norm_mse": sum_norm_mse / n,
        "raw_mse": sum_raw_mse / n,
        "raw_sign": sum_raw_sign / n
    }


# -------------------------------------------------
# val epoch
# -------------------------------------------------
@torch.no_grad()
def val_epoch(model, loader, device_conv):

    model.eval()

    sum_norm_mse = 0
    sum_raw_mse = 0
    sum_raw_sign = 0
    n = 0

    for x, y, cfg, bdir in loader:

        x = x.to(device_conv)

        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        y_norm = (y - y_mean) / (y_std + 1e-6)

        pred_norm, aux = model(x)

        norm_mse = torch.mean((pred_norm - y_norm) ** 2)

        pred_raw = pred_norm * y_std + y_mean

        raw_metrics = compute_metrics(pred_raw, y)

        sum_norm_mse += float(norm_mse)
        sum_raw_mse += raw_metrics["mse"]
        sum_raw_sign += raw_metrics["sign"]

        n += 1

    return {
        "norm_mse": sum_norm_mse / n,
        "raw_mse": sum_raw_mse / n,
        "raw_sign": sum_raw_sign / n
    }


# -------------------------------------------------
# train
# -------------------------------------------------
def train():

    set_seed()

    # -------------------------
    # device
    # -------------------------
    if torch.cuda.is_available():
        print("[GPU]", torch.cuda.get_device_name(0))
        device_conv = torch.device("cuda")
    else:
        device_conv = torch.device("cpu")

    device_fc = torch.device("cpu")

    # -------------------------
    # dataset
    # -------------------------
    dataset = OAI_Bundle_Dataset(
        "/home/richard93513/SpikingRx-on-OAI/spx_records/bundle"
    )

    n_total = len(dataset)
    n_val = int(n_total * 0.15)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(1234)
    )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    print(f"[Dataset] total={n_total} train={n_train} val={n_val}")

    # -------------------------
    # model
    # -------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        out_bits=14400,
        T=1,
        device_conv=device_conv,
        device_fc=device_fc,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # -------------------------
    # training config
    # -------------------------
    max_epochs = 5
    patience = 10

    best_val = float("inf")
    no_improve = 0

    log_path = os.path.join(CURRENT_DIR, "train_log_norm.txt")

    with open(log_path, "w") as f:
        f.write("epoch train_norm_mse val_norm_mse train_raw_mse val_raw_mse train_sign val_sign\n")

    epoch_list = []
    train_curve = []
    val_curve = []

    # -------------------------
    # loop
    # -------------------------
    for ep in range(1, max_epochs + 1):

        print(f"\n===== Epoch {ep} =====")

        train_stats = train_epoch(model, train_loader, optimizer, device_conv)
        val_stats = val_epoch(model, val_loader, device_conv)

        print(
            f"[Train] norm_mse={train_stats['norm_mse']:.4f} "
            f"raw_mse={train_stats['raw_mse']:.2f} "
            f"sign={train_stats['raw_sign']:.3f}"
        )

        print(
            f"[Val]   norm_mse={val_stats['norm_mse']:.4f} "
            f"raw_mse={val_stats['raw_mse']:.2f} "
            f"sign={val_stats['raw_sign']:.3f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{ep} "
                f"{train_stats['norm_mse']} "
                f"{val_stats['norm_mse']} "
                f"{train_stats['raw_mse']} "
                f"{val_stats['raw_mse']} "
                f"{train_stats['raw_sign']} "
                f"{val_stats['raw_sign']}\n"
            )

        epoch_list.append(ep)
        train_curve.append(train_stats["norm_mse"])
        val_curve.append(val_stats["norm_mse"])

        plot_curve(
            epoch_list,
            train_curve,
            val_curve,
            "Normalized MSE",
            "MSE",
            os.path.join(CURRENT_DIR, "curve_norm_mse.png")
        )

        # early stopping
        if val_stats["norm_mse"] < best_val:

            best_val = val_stats["norm_mse"]
            no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(CURRENT_DIR, "best_spikingrx_model_norm.pth")
            )

            print("✓ new best model")

        else:

            no_improve += 1
            print(f"no improve {no_improve}/{patience}")

        if no_improve >= patience:
            print("EARLY STOP")
            break


if __name__ == "__main__":
    train()
