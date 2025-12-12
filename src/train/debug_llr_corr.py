import torch
import numpy as np
from data.dataset_oai_bundle import OAI_Bundle_Dataset
from models.spikingrx_model import SpikingRxModel

bundle_root = "/home/richard93513/SpikingRx-on-OAI/spx_records/bundle"
ckpt_path   = "/home/richard93513/SpikingRx-on-OAI/src/train/best_spikingrx_model.pth"

# 1) 準備 dataset & 隨便取一筆
ds = OAI_Bundle_Dataset(bundle_root=bundle_root, limit=1)
x, y_llr, cfg, bdir = ds[0]   # x: [3,2,32,32], y_llr: [G]

# 2) 準備模型（要跟 train / inference 完全同一版）
device_conv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_fc   = torch.device("cpu")

model = SpikingRxModel(
    in_ch=2,
    base_ch=16,
    bits_per_symbol=2,
    beta=0.9,
    theta=0.5,
    llr_temperature=1.0,
    out_bits=y_llr.numel(),   # 14400
    T=3,
    device_conv=device_conv,
    device_fc=device_fc,
)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# 3) forward 一次
x = x.unsqueeze(0)                  # [1,3,2,32,32]
with torch.no_grad():
    y_pred, aux = model(x)

y_pred = y_pred.squeeze(0).cpu().numpy()  # [G]
y_true = y_llr.numpy()                    # [G]

print("pred  min/max/mean/std =", y_pred.min(), y_pred.max(), y_pred.mean(), y_pred.std())
print("true  min/max/mean/std =", y_true.min(), y_true.max(), y_true.mean(), y_true.std())

# 4) 算 Pearson correlation
corr = np.corrcoef(y_pred, y_true)[0,1]
print("Pearson corr(pred, true) =", corr)

# 5) 看 sign 一致的比例（忽略非常接近 0 的）
eps = 1e-3
mask = np.abs(y_true) > eps
same_sign = np.mean( np.sign(y_pred[mask]) == np.sign(y_true[mask]) )
print("Sign match ratio =", same_sign)

