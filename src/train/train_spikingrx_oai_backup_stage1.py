# src/train/train_spikingrx_oai.py
# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Reproducibility
# -------------------------------------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Plot utils
# -------------------------------------------------
def plot_curves(epoch_list, train_list, val_list, ylabel, title, out_path):
    if len(epoch_list) == 0:
        return

    plt.figure()
    plt.plot(epoch_list, train_list, marker="o", label="train")
    plt.plot(epoch_list, val_list, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------
# Loss
# -------------------------------------------------
class LLRLoss(nn.Module):
    """
    Combined loss for LLR regression.

    total_loss =
        lambda_l1     * SmoothL1(pred, target)
      + lambda_cosine * (1 - cosine_similarity(pred, target))
    """

    def __init__(self, lambda_l1=1.0, lambda_cosine=0.1, beta=1.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_cosine = lambda_cosine
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred, target):
        l1 = self.smooth_l1(pred, target)

        # cosine over feature dimension
        cos = F.cosine_similarity(pred, target, dim=1).mean()
        cos_loss = 1.0 - cos

        total = self.lambda_l1 * l1 + self.lambda_cosine * cos_loss

        stats = {
            "smooth_l1": float(l1.detach().cpu()),
            "cosine_loss": float(cos_loss.detach().cpu()),
            "cosine_sim": float(cos.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total, stats


# -------------------------------------------------
# Metrics
# -------------------------------------------------
@torch.no_grad()
def compute_batch_metrics(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    sign_acc = ((pred >= 0) == (target >= 0)).float().mean()

    return {
        "mse": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "sign_acc": float(sign_acc.detach().cpu()),
    }


# -------------------------------------------------
# One epoch: train
# -------------------------------------------------
def run_train_epoch(model, loader, optimizer, loss_fn, spk_device, label_device, grad_clip=1.0):
    model.train()

    sum_loss = 0.0
    sum_mse = 0.0
    sum_mae = 0.0
    sum_sign = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", ncols=110)

    for x, y_llr, cfg, bdir in pbar:
        # x: [B,T,2,32,32]
        x = x.to(spk_device)
        y_llr = y_llr.to(label_device)

        optimizer.zero_grad()

        y_pred, aux = model(x)   # y_pred on CPU
        loss, loss_stats = loss_fn(y_pred, y_llr)

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_metrics = compute_batch_metrics(y_pred, y_llr)

        sum_loss += float(loss.detach().cpu())
        sum_mse += batch_metrics["mse"]
        sum_mae += batch_metrics["mae"]
        sum_sign += batch_metrics["sign_acc"]
        n_batches += 1

        pbar.set_postfix(
            loss=f'{float(loss.detach().cpu()):.4f}',
            mse=f'{batch_metrics["mse"]:.2f}',
            sign=f'{batch_metrics["sign_acc"]:.3f}'
        )

    avg = {
        "loss": sum_loss / max(n_batches, 1),
        "mse": sum_mse / max(n_batches, 1),
        "mae": sum_mae / max(n_batches, 1),
        "sign_acc": sum_sign / max(n_batches, 1),
    }
    return avg


# -------------------------------------------------
# One epoch: val
# -------------------------------------------------
@torch.no_grad()
def run_val_epoch(model, loader, loss_fn, spk_device, label_device):
    model.eval()

    sum_loss = 0.0
    sum_mse = 0.0
    sum_mae = 0.0
    sum_sign = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Val  ", ncols=110)

    for x, y_llr, cfg, bdir in pbar:
        x = x.to(spk_device)
        y_llr = y_llr.to(label_device)

        y_pred, aux = model(x)
        loss, loss_stats = loss_fn(y_pred, y_llr)
        batch_metrics = compute_batch_metrics(y_pred, y_llr)

        sum_loss += float(loss.detach().cpu())
        sum_mse += batch_metrics["mse"]
        sum_mae += batch_metrics["mae"]
        sum_sign += batch_metrics["sign_acc"]
        n_batches += 1

        pbar.set_postfix(
            loss=f'{float(loss.detach().cpu()):.4f}',
            mse=f'{batch_metrics["mse"]:.2f}',
            sign=f'{batch_metrics["sign_acc"]:.3f}'
        )

    avg = {
        "loss": sum_loss / max(n_batches, 1),
        "mse": sum_mse / max(n_batches, 1),
        "mae": sum_mae / max(n_batches, 1),
        "sign_acc": sum_sign / max(n_batches, 1),
    }
    return avg


def train():
    set_seed(1234)

    # -----------------------------
    # Output paths
    # -----------------------------
    log_path = os.path.join(CURRENT_DIR, "train_log.txt")
    ckpt_best_path = os.path.join(CURRENT_DIR, "best_spikingrx_model.pth")
    ckpt_last_path = os.path.join(CURRENT_DIR, "last_spikingrx_model.pth")

    fig_loss_path = os.path.join(CURRENT_DIR, "curve_loss.png")
    fig_mse_path = os.path.join(CURRENT_DIR, "curve_mse.png")
    fig_sign_path = os.path.join(CURRENT_DIR, "curve_sign_acc.png")

    # -----------------------------
    # Hyperparams
    # -----------------------------
    batch_size = 1
    max_epochs = 3
    lr = 1e-4
    weight_decay = 1e-6
    val_ratio = 0.15
    patience = 10
    grad_clip = 1.0

    # loss weights
    lambda_l1 = 1.0
    lambda_cosine = 0.1

    # -----------------------------
    # Device
    # -----------------------------
    if torch.cuda.is_available():
        print("[Device] CUDA GPU =", torch.cuda.get_device_name(0))
        spk_device = torch.device("cuda")
    else:
        print("[Device] CPU only")
        spk_device = torch.device("cpu")

    fc_device = torch.device("cpu")
    label_device = torch.device("cpu")

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = OAI_Bundle_Dataset(
        bundle_root="/home/richard93513/SpikingRx-on-OAI/spx_records/bundle",
        limit=None,
        normalize=True,
    )

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val

    split_gen = torch.Generator().manual_seed(1234)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=split_gen)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"[Dataset] total={n_total}, train={n_train}, val={n_val}")

    # -----------------------------
    # Model
    # -----------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
        out_bits=14400,
        T=3,
        device_conv=spk_device,
        device_fc=fc_device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = LLRLoss(
        lambda_l1=lambda_l1,
        lambda_cosine=lambda_cosine,
        beta=1.0,
    )

    # -----------------------------
    # Logging init
    # -----------------------------
    with open(log_path, "w") as f:
        f.write("# SpikingRx-on-OAI training log\n")
        f.write(f"# start_time = {datetime.datetime.now().isoformat()}\n")
        f.write(f"# total={n_total}, train={n_train}, val={n_val}\n")
        f.write(f"# lr={lr}, batch_size={batch_size}, max_epochs={max_epochs}\n")
        f.write(f"# loss = SmoothL1 + {lambda_cosine} * cosine_loss\n")
        f.write(
            "epoch\t"
            "train_loss\ttrain_mse\ttrain_mae\ttrain_sign_acc\t"
            "val_loss\tval_mse\tval_mae\tval_sign_acc\n"
        )

    epoch_hist = []
    train_loss_hist = []
    val_loss_hist = []
    train_mse_hist = []
    val_mse_hist = []
    train_sign_hist = []
    val_sign_hist = []

    # -----------------------------
    # Early stopping
    # -----------------------------
    best_val_loss = float("inf")
    no_improve = 0

    print("\n[Start Training]")
    print(f"  max_epochs   = {max_epochs}")
    print(f"  log_path     = {log_path}")
    print(f"  best_ckpt    = {ckpt_best_path}")
    print(f"  last_ckpt    = {ckpt_last_path}")
    print("")

    for ep in range(1, max_epochs + 1):
        print(f"\n================ EPOCH {ep}/{max_epochs} ================")

        train_stats = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            spk_device=spk_device,
            label_device=label_device,
            grad_clip=grad_clip,
        )

        val_stats = run_val_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            spk_device=spk_device,
            label_device=label_device,
        )

        print(
            "[Train] "
            f"loss={train_stats['loss']:.6f}  "
            f"mse={train_stats['mse']:.6f}  "
            f"mae={train_stats['mae']:.6f}  "
            f"sign_acc={train_stats['sign_acc']:.6f}"
        )

        print(
            "[Val]   "
            f"loss={val_stats['loss']:.6f}  "
            f"mse={val_stats['mse']:.6f}  "
            f"mae={val_stats['mae']:.6f}  "
            f"sign_acc={val_stats['sign_acc']:.6f}"
        )

        # append history
        epoch_hist.append(ep)
        train_loss_hist.append(train_stats["loss"])
        val_loss_hist.append(val_stats["loss"])
        train_mse_hist.append(train_stats["mse"])
        val_mse_hist.append(val_stats["mse"])
        train_sign_hist.append(train_stats["sign_acc"])
        val_sign_hist.append(val_stats["sign_acc"])

        # write log
        with open(log_path, "a") as f:
            f.write(
                f"{ep}\t"
                f"{train_stats['loss']:.6f}\t"
                f"{train_stats['mse']:.6f}\t"
                f"{train_stats['mae']:.6f}\t"
                f"{train_stats['sign_acc']:.6f}\t"
                f"{val_stats['loss']:.6f}\t"
                f"{val_stats['mse']:.6f}\t"
                f"{val_stats['mae']:.6f}\t"
                f"{val_stats['sign_acc']:.6f}\n"
            )

        # save last checkpoint every epoch
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_stats": train_stats,
                "val_stats": val_stats,
            },
            ckpt_last_path,
        )

        # plot curves
        plot_curves(
            epoch_hist, train_loss_hist, val_loss_hist,
            ylabel="Loss",
            title="Train/Val Loss",
            out_path=fig_loss_path,
        )
        plot_curves(
            epoch_hist, train_mse_hist, val_mse_hist,
            ylabel="MSE",
            title="Train/Val MSE",
            out_path=fig_mse_path,
        )
        plot_curves(
            epoch_hist, train_sign_hist, val_sign_hist,
            ylabel="Sign Accuracy",
            title="Train/Val Sign Accuracy",
            out_path=fig_sign_path,
        )

        # early stopping by val loss
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            no_improve = 0

            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                },
                ckpt_best_path,
            )
            print(f"[Best] improved val_loss -> {best_val_loss:.6f}")
        else:
            no_improve += 1
            print(f"[EarlyStop] no improvement: {no_improve}/{patience}")

        if no_improve >= patience:
            print("\n>>> EARLY STOPPING — stopped by validation patience.")
            break

    print("\nTraining finished.")
    print("Best checkpoint:", ckpt_best_path)
    print("Last checkpoint:", ckpt_last_path)
    print("Curves:")
    print(" ", fig_loss_path)
    print(" ", fig_mse_path)
    print(" ", fig_sign_path)


if __name__ == "__main__":
    train()
