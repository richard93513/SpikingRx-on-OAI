# src/train/train_spikingrx_oai.py
# -*- coding: utf-8 -*-

import os
import sys
import random
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


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def compute_metrics(logits, target_llr):
    pred_sign = (logits >= 0)
    true_sign = (target_llr >= 0)

    sign_acc = (pred_sign == true_sign).float().mean()

    # 額外觀察：把 logits 當成 pseudo-LLR 時，和 target 的 MSE
    mse = torch.mean((logits - target_llr) ** 2)
    mae = torch.mean(torch.abs(logits - target_llr))

    return {
        "mse": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "sign": float(sign_acc.detach().cpu()),
    }


def train_epoch(model, loader, optimizer, loss_fn, device_model):
    model.train()

    sum_loss = 0.0
    sum_raw_mse = 0.0
    sum_raw_sign = 0.0
    n = 0

    pbar = tqdm(loader, ncols=110, leave=False, dynamic_ncols=True)

    for x, y, cfg, bdir in pbar:
        x = x.to(device_model)
        y = y.to(device_model)

        # sign target: LLR >= 0 -> 1, else 0
        y_sign = (y >= 0).float()

        logits, aux = model(x)
        logits = logits.to(device_model)

        loss = loss_fn(logits, y_sign)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        raw_metrics = compute_metrics(logits, y)

        sum_loss += float(loss.detach().cpu())
        sum_raw_mse += raw_metrics["mse"]
        sum_raw_sign += raw_metrics["sign"]
        n += 1

        pbar.set_postfix_str(f"bce={float(loss.detach().cpu()):.3f} sign={raw_metrics['sign']:.3f}")

    return {
        "loss": sum_loss / n,
        "raw_mse": sum_raw_mse / n,
        "raw_sign": sum_raw_sign / n,
    }


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device_model):
    model.eval()

    sum_loss = 0.0
    sum_raw_mse = 0.0
    sum_raw_sign = 0.0
    n = 0

    for x, y, cfg, bdir in loader:
        x = x.to(device_model)
        y = y.to(device_model)

        y_sign = (y >= 0).float()

        logits, aux = model(x)
        logits = logits.to(device_model)

        loss = loss_fn(logits, y_sign)
        raw_metrics = compute_metrics(logits, y)

        sum_loss += float(loss.detach().cpu())
        sum_raw_mse += raw_metrics["mse"]
        sum_raw_sign += raw_metrics["sign"]
        n += 1

    return {
        "loss": sum_loss / n,
        "raw_mse": sum_raw_mse / n,
        "raw_sign": sum_raw_sign / n,
    }


def train():
    set_seed()

    if torch.cuda.is_available():
        print("[GPU]", torch.cuda.get_device_name(0))
        device_model = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device_model = torch.device("cpu")

    dataset = OAI_Bundle_Dataset(
        "/home/richard93513/SpikingRx-on-OAI/spx_records/bundle",
        T=1,
        keep_rect=True,
        normalize=True,
    )

    x0, y0, cfg0, bdir0 = dataset[0]
    print("[Sample] bdir =", bdir0)
    print("[Sample] x.shape =", tuple(x0.shape))
    print("[Sample] y.shape =", tuple(y0.shape))
    print("[Sample] x.mean/std =", float(x0.mean()), float(x0.std()))
    print("[Sample] y.mean/std =", float(y0.mean()), float(y0.std()))

    n_total = len(dataset)
    n_val = int(n_total * 0.15)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(1234),
    )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    print(f"[Dataset] total={n_total} train={n_train} val={n_val}")

    model = SpikingRxModel(
        in_ch=4,
        base_ch=8,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        out_bits=14400,
        T=1,
        device_conv=device_model,
        device_fc=device_model,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    max_epochs = 30
    patience = 5

    best_val = float("inf")
    no_improve = 0

    log_path = os.path.join(CURRENT_DIR, "train_log_rect_sign_bce.txt")
    with open(log_path, "w") as f:
        f.write("epoch train_bce val_bce train_raw_mse val_raw_mse train_sign val_sign\n")

    epoch_list = []
    train_curve = []
    val_curve = []

    for ep in range(1, max_epochs + 1):
        print(f"\n===== Epoch {ep} =====")

        train_stats = train_epoch(model, train_loader, optimizer, loss_fn, device_model)
        val_stats = val_epoch(model, val_loader, loss_fn, device_model)

        print(
            f"[Train] bce={train_stats['loss']:.4f} "
            f"raw_mse={train_stats['raw_mse']:.2f} "
            f"sign={train_stats['raw_sign']:.3f}"
        )
        print(
            f"[Val]   bce={val_stats['loss']:.4f} "
            f"raw_mse={val_stats['raw_mse']:.2f} "
            f"sign={val_stats['raw_sign']:.3f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{ep} "
                f"{train_stats['loss']} "
                f"{val_stats['loss']} "
                f"{train_stats['raw_mse']} "
                f"{val_stats['raw_mse']} "
                f"{train_stats['raw_sign']} "
                f"{val_stats['raw_sign']}\n"
            )

        epoch_list.append(ep)
        train_curve.append(train_stats["loss"])
        val_curve.append(val_stats["loss"])

        plot_curve(
            epoch_list,
            train_curve,
            val_curve,
            "Rectangular Sign-BCE",
            "BCE",
            os.path.join(CURRENT_DIR, "curve_rect_sign_bce.png"),
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(CURRENT_DIR, "best_spikingrx_model_rect_sign_bce.pth"),
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
