# src/train/train_spikingrx_oai.py
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
# 繪圖工具：每個 epoch 更新一次 train_loss.png
# -------------------------------------------------
def plot_loss_curve(epoch_list, loss_list, out_path):
    if len(epoch_list) == 0:
        return

    plt.figure()
    plt.plot(epoch_list, loss_list, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total training loss (sum over batches)")
    plt.title("SpikingRx-on-OAI Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train():

    # -----------------------------
    #  Log / 圖檔路徑
    # -----------------------------
    log_path = os.path.join(CURRENT_DIR, "train_log.txt")
    fig_path = os.path.join(CURRENT_DIR, "train_loss.png")

    # 先清空舊的 log（如果想保留歷史，可以改成 "a" 並不覆寫）
    with open(log_path, "w") as f:
        f.write("# SpikingRx-on-OAI training log\n")
        f.write("# start_time = {}\n".format(datetime.datetime.now().isoformat()))
        f.write("# columns: epoch  total_loss\n")

    epoch_history = []
    loss_history = []

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
        limit=None,   # 如果要先小規模試，可以改成小一點的整數
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # -----------------------------
    # Model（spiking GPU + readout CPU）
    # -----------------------------
    model = SpikingRxModel(
        in_ch=2,
        base_ch=16,
        bits_per_symbol=2,
        beta=0.9,
        theta=0.5,
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
    patience = 8            # 比原本 2 大很多，讓它有時間慢慢收斂
    best_loss = float("inf")
    no_improve = 0
    best_path = os.path.join(CURRENT_DIR, "best_spikingrx_model.pth")

    max_epochs = 80
    print(f"\n[Start Training] max_epochs = {max_epochs}")
    print(f"[LOG] loss log → {log_path}")
    print(f"[LOG] loss figure → {fig_path}\n")

    # -----------------------------
    # Training Loop
    # -----------------------------
    for ep in range(1, max_epochs + 1):
        model.train()   # 確保在 train 模式
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {ep}", ncols=100)

        for x, y_llr, cfg, bdir in pbar:
            # -------------------------
            # 準備資料
            # -------------------------
            # x: [B, T, 2, 32, 32]，先丟到 spiking device（GPU 或 CPU）
            x = x.to(spk_device)

            # labels 在 CPU
            y_llr = y_llr.to(label_device)   # [B, G]，這裡 B=1

            # -------------------------
            # forward
            # -------------------------
            y_pred, aux = model(x)   # hybrid forward; y_pred 在 CPU（readoutANN）

            # -------------------------
            # loss
            # -------------------------
            loss = loss_fn(y_pred, y_llr)

            opt.zero_grad()
            loss.backward()

            # 避免梯度爆掉，稍微 clip 一下
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.3f}")

        # -----------------------------
        # Epoch 結束 → 紀錄 / 畫圖 / early stopping
        # -----------------------------
        print(f"\n>>> Epoch {ep} Total Loss = {total_loss:.4f}")

        # 寫入 log 檔
        with open(log_path, "a") as f:
            f.write(f"{ep}\t{total_loss:.6f}\n")

        # 更新 history 並畫圖
        epoch_history.append(ep)
        loss_history.append(total_loss)
        plot_loss_curve(epoch_history, loss_history, fig_path)

        # Early stopping 判斷
        if total_loss < best_loss:
            best_loss = total_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f" ✓ Improved — saved best model ({best_loss:.4f})")
        else:
            no_improve += 1
            print(f" ✗ No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print("\n>>> EARLY STOPPING — Training Ended by patience.")
            break

    print("\nTraining finished.")
    print("Best model saved at:", best_path)


if __name__ == "__main__":
    train()

