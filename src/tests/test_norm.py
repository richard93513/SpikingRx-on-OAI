# src/tests/test_norm.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

import torch
from norm_layer import SpikeNorm

# --------------------------------------------------------
# 測試 SpikeNorm 模組
# --------------------------------------------------------
norm = SpikeNorm(num_channels=8)
x = torch.randn(2, 8, 32, 32)     # 單時間步輸入
x_seq = torch.randn(2, 5, 8, 32, 32)  # 多時間步輸入

print("=== SpikeNorm Forward Test ===")

# 單步測試
y = norm(x)
print("Input shape :", x.shape)
print("Output shape:", y.shape)
print("Output mean :", y.mean().item(), "std:", y.std().item())

# 多步測試
y_seq = norm(x_seq)
print("\n=== SpikeNorm Temporal Test ===")
print("Input shape :", x_seq.shape)
print("Output shape:", y_seq.shape)
print("Mean/std at t=0:", y_seq[:,0].mean().item(), y_seq[:,0].std().item())
print("Mean/std at t=4:", y_seq[:,4].mean().item(), y_seq[:,4].std().item())

import matplotlib.pyplot as plt

# --------------------------------------------------------
# 額外視覺化：顯示正規化前後的分布
# --------------------------------------------------------
x_before = x_seq.detach().cpu().numpy().ravel()      # 原始輸入分布
x_after = y_seq.detach().cpu().numpy().ravel()       # 正規化後分布

plt.figure(figsize=(6, 4))
plt.hist(x_before, bins=80, alpha=0.5, label='Before Norm')
plt.hist(x_after, bins=80, alpha=0.5, label='After Norm')
plt.legend()
plt.title("SpikeNorm Normalization Effect")
plt.xlabel("Value")
plt.ylabel("Frequency")

save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/spikenorm_distribution.png"))
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path)
plt.close()

print(f"✅ 分布圖已儲存到: {save_path}")




