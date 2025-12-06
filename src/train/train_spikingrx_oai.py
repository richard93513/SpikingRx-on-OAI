# src/train/train_spikingrx_oai.py
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
DATA_DIR = os.path.join(SRC_DIR, "data")

for p in [SRC_DIR, DATA_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from data.dataset_oai_bundle import OAI_Bundle_Dataset
from models.spikingrx_model import SpikingRxModel


def train():

    # -----------------------------
    #  Device 選擇
    # -----------------------------
    if torch.cuda.is_available():
        print("[Device] CUDA GPU =", torch.cuda.get_device_name(0))
        spk_device = torch.device("cuda")   # Spiking Blocks 在 GPU
    else:
        print("[Device] CPU only")
        spk_device = torch.device("cpu")

    fc_device = torch.device("cpu")         # ReadoutANN 固定 CPU
    label_device = torch.device("cpu")      # LLR label 也放 CPU

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = OAI_Bundle_Dataset(
        bundle_root="/home/richard93513/SpikingRx-on-OAI/spx_records/bundle",
        limit=None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # -----------------------------
    # Model（spiking GPU + readout CPU）
    # -----------------------------
    model = SpikingRxModel(
        in_ch=2, base_ch=16,
        bits_per_symbol=2,
        beta=0.9, theta=0.5,
        llr_temperature=1.0,
        out_bits=14400,
        device_conv=spk_device,
        device_fc=fc_device,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # -----------------------------
    # Early Stopping
    # -----------------------------
    patience = 2
    best_loss = float("inf")
    no_improve = 0
    best_path = os.path.join(CURRENT_DIR, "best_spikingrx_model.pth")

    max_epochs = 30
    print(f"\n[Start Training] max_epochs = {max_epochs}")

    # -----------------------------
    # Training Loop
    # -----------------------------
    for ep in range(1, max_epochs + 1):
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {ep}", ncols=100)

        for x, y_llr, cfg, bdir in pbar:

            # Spiking 前端 → GPU（如果有）
            x = x.to(spk_device)

            # labels → CPU
            y_llr = y_llr.to(label_device)

            # forward
            y_pred, _ = model(x)   # hybrid forward

            # loss 只能在 CPU（因為 readout 是 CPU）
            loss = loss_fn(y_pred, y_llr)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # -----------------------------
        # Epoch 結束 → early stopping
        # -----------------------------
        print(f"\n>>> Epoch {ep} Total Loss = {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f" ✓ Improved — saved best model ({best_loss:.4f})")
        else:
            no_improve += 1
            print(f" ✗ No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print("\n>>> EARLY STOPPING — Training Ended.")
            break

    print("\nTraining finished.")
    print("Best model saved at:", best_path)


if __name__ == "__main__":
    train()



