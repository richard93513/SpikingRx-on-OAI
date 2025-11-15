# src/tests/test_lif.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

import torch
import matplotlib.pyplot as plt
from lif_neuron import LIF

# --------------------------------------------------------
# 建立 LIF 模組
# --------------------------------------------------------
lif = LIF(beta=0.9, theta=0.5)

# 一維特徵測試
B, T, D = 1, 20, 1
I = torch.ones(B, T, D) * 1.0 + 0.1 * torch.randn(B, T, D)

print("=== Test A: [B,T,D] ===")
print("Input shape:", I.shape)
out = lif(I)
print("Output shape:", out.shape)
print("Spikes over time:", out.view(T))
print("Total spikes:", int(out.sum().item()))

# --------------------------------------------------------
# 額外記錄膜電位 U[t]、輸入 I[t]、spike S[t] 並畫圖
# --------------------------------------------------------
U = torch.zeros(B, D)
U_list, S_list, I_list = [], [], []

for t in range(T):
    It = I[:, t]
    U = lif.beta * U + (1 - lif.beta) * It
    S = (U >= lif.theta).float()
    U = U - S * lif.theta

    I_list.append(It.item())
    U_list.append(U.item())
    S_list.append(S.item())

# --------------------------------------------------------
# 畫圖
# --------------------------------------------------------
time = list(range(1, T + 1))
fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(time, I_list, 'k--', label='Input I[t]', alpha=0.5)
ax1.plot(time, U_list, 'b-', label='Membrane Potential U[t]')
ax1.axhline(y=lif.theta.item(), color='r', linestyle=':', label='Threshold θ')
ax1.set_xlabel('Time step')
ax1.set_ylabel('U[t], I[t]')
ax1.legend(loc='upper left')

# Spike (用另一軸畫 0/1)
ax2 = ax1.twinx()
ax2.scatter(time, S_list, color='orange', label='Spike S[t]', marker='o')
ax2.set_ylabel('Spike (0 or 1)')
ax2.set_ylim(-0.1, 1.2)
ax2.legend(loc='upper right')

plt.title('LIF Neuron Dynamics')
plt.tight_layout()

# --------------------------------------------------------
# 儲存圖檔到 data/
# --------------------------------------------------------
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/lif_spike_plot.png"))
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close()

print(f"✅ 圖片已儲存到: {save_path}")



