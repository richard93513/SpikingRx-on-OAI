# src/tests/test_conv.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

import torch
import matplotlib.pyplot as plt
from conv_block import ConvBlock

# --------------------------------------------------------
# 建立並測試 ConvBlock
# --------------------------------------------------------
conv = ConvBlock(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
x = torch.randn(2, 3, 32, 32)   # 模擬輸入 [B,C,H,W]

print("=== ConvBlock Forward Test ===")
print("Input shape :", x.shape)
y = conv(x)
print("Output shape:", y.shape)
print(f"Output sample (mean,std): {y.mean().item():.4f}, {y.std().item():.4f}")

# --------------------------------------------------------
# 一、視覺化卷積核 (filters)
# --------------------------------------------------------
weights = conv.conv.weight.data.clone().detach().cpu()
out_channels, in_channels, kh, kw = weights.shape
print(f"Conv filters shape: {weights.shape}")

fig, axes = plt.subplots(out_channels, in_channels, figsize=(in_channels*2, out_channels*2))
fig.suptitle("Conv Filters", fontsize=14)

for i in range(out_channels):
    for j in range(in_channels):
        ax = axes[i, j] if out_channels > 1 else axes[j]
        w = weights[i, j].numpy()
        ax.imshow(w, cmap='bwr', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"f{i}-ch{j}")
plt.tight_layout()

save_path_filters = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/conv_filters.png"))
os.makedirs(os.path.dirname(save_path_filters), exist_ok=True)
plt.savefig(save_path_filters)
plt.close()
print(f"✅ 卷積核圖已儲存到: {save_path_filters}")

# --------------------------------------------------------
# 二、視覺化卷積後的特徵圖 (feature maps)
# --------------------------------------------------------
with torch.no_grad():
    feature_maps = conv(x)[0]  # 取第1筆輸出 [C,H,W]

fig, axes = plt.subplots(1, feature_maps.shape[0], figsize=(feature_maps.shape[0]*2, 2))
fig.suptitle("Output Feature Maps", fontsize=14)

for i, ax in enumerate(axes):
    fm = feature_maps[i].detach().cpu().numpy()
    ax.imshow(fm, cmap='viridis')
    ax.axis('off')
    ax.set_title(f"Map {i}")
plt.tight_layout()

save_path_fm = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/conv_featuremaps.png"))
plt.savefig(save_path_fm)
plt.close()
print(f"✅ 特徵圖已儲存到: {save_path_fm}")


