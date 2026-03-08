# src/tests/test_sew.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import torch
from models.sew_block import SEWBlock


# --------------------------------------------------------
# Helper → print 2D slice (8×8)
# --------------------------------------------------------
def print_matrix(title, x, max_h=8, max_w=8):
    print(f"\n--- {title} ({max_h}x{max_w}) ---")
    print(x[:max_h, :max_w])


# --------------------------------------------------------
# Create SEW block
# --------------------------------------------------------
in_channels = 4
out_channels = 8
sew = SEWBlock(in_channels, out_channels)


# --------------------------------------------------------
# Prepare fake OFDM-like data [B,T,C,H,W]
# --------------------------------------------------------
B, T, C, H, W = 1, 5, in_channels, 32, 32
x = torch.randn(B, T, C, H, W)

print("\n======================================")
print("    ORIGINAL OFDM-LIKE INPUT SLICES    ")
print("======================================")
for t in range(4):
    for c in range(4):
        print_matrix(f"Input t={t}, c={c}", x[0, t, c])


# --------------------------------------------------------
# Manual SEWBlock debug: Conv / Norm / LIF / Shortcut
# --------------------------------------------------------
conv = sew.conv
norm = sew.norm
lif  = sew.lif
shortcut_layer = sew.shortcut

print("\n======================================")
print("          DEBUG: CONV OUTPUT          ")
print("======================================")

conv_out_all = []
for t in range(4):
    xt = x[0, t].unsqueeze(0)
    conv_out = conv(xt)
    conv_out_all.append(conv_out)

    for c in range(4):
        print_matrix(f"Conv t={t}, c={c}", conv_out[0, c])


print("\n======================================")
print("          DEBUG: NORM OUTPUT          ")
print("======================================")

norm_out_all = []
for t in range(4):
    norm_out = norm(conv_out_all[t])
    norm_out_all.append(norm_out)

    for c in range(4):
        print_matrix(f"Norm t={t}, c={c}", norm_out[0, c])


print("\n======================================")
print("           DEBUG: LIF SPIKES          ")
print("======================================")

lif_out_all = []
for t in range(4):
    # LIF expects [B,T,C,H,W]
    lif_input = norm_out_all[t].unsqueeze(1)
    lif_out = lif(lif_input)       # → [B,1,C,H,W]
    spikes = lif_out[:, 0]
    lif_out_all.append(spikes)

    for c in range(4):
        print_matrix(f"LIF spikes t={t}, c={c}", spikes[0, c])


print("\n======================================")
print("        DEBUG: SHORTCUT OUTPUT        ")
print("======================================")

shortcut_all = []
for t in range(4):
    xt = x[0, t].unsqueeze(0)
    sc = shortcut_layer(xt)
    shortcut_all.append(sc)

    for c in range(4):
        print_matrix(f"Shortcut t={t}, c={c}", sc[0, c])


print("\n======================================")
print("        FINAL SEW OUTPUT SLICES       ")
print("======================================")

# Final output must match SEW forward behavior
final_output_all = []
for t in range(4):
    y = lif_out_all[t] + shortcut_all[t]
    final_output_all.append(y)

    for c in range(4):
        print_matrix(f"SEW output t={t}, c={c}", y[0, c])


print("\nAll debug slices printed successfully. 🚀")





