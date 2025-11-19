# src/train/train_spikingrx.py
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from train.dataset_simple_ofdm import SimpleOFDMDataset
from models.spikingrx_model import SpikingRxModel


def plot_loss(loss_list, out_png):
    fig, ax = plt.subplots()
    ax.plot(loss_list, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print("Saved:", out_png)


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = SimpleOFDMDataset(num_samples=10000, snr_db=20)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
        llr_temperature=1.0,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 20
    loss_history = []

    for ep in range(1, epochs + 1):
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            opt.zero_grad()

            llr, aux = model(batch_x)
            loss = loss_fn(llr, batch_y)

            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"[Epoch {ep:02d}] Loss = {total_loss:.4f}")
        loss_history.append(total_loss)

    # ------------------------------
    # 儲存模型
    # ------------------------------
    ckpt_path = os.path.join(CURRENT_DIR, "spikingrx_checkpoint.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Checkpoint saved:", ckpt_path)

    # ------------------------------
    # 儲存 loss history
    # ------------------------------
    plot_dir = os.path.join(CURRENT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    loss_png = os.path.join(plot_dir, "loss_curve.png")
    plot_loss(loss_history, loss_png)

    np.save(os.path.join(plot_dir, "loss_history.npy"), np.array(loss_history))
    print("Saved: loss_history.npy")

    with open(os.path.join(plot_dir, "train_log.txt"), "w") as f:
        for i, l in enumerate(loss_history):
            f.write(f"Epoch {i+1}: {l}\n")


if __name__ == "__main__":
    train()


