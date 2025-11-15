# src/models/lif_neuron.py

# --------------------------------------------------------
# 模組位置對應：SpikingRx → SEW-ResNet block → LIF 神經元
# --------------------------------------------------------
# 這個檔案 lif_neuron.py 對應 SpikingRx 論文架構中的「spiking activation 層」
# 在 SEW-ResNet block 中的流程為：
# Conv → Norm → LIF(spike產生) → Shortcut → 下一層
# LIF 是在這裡把連續值 feature 轉成 0/1 的 spike。
# --------------------------------------------------------

import torch                    # 匯入 PyTorch 主函式庫
import torch.nn as nn           # 匯入神經網路模組 (nn.Module, Parameter)
# --------------------------------------------------------
# SpikingRx: 實際訓練與推論都是基於 PyTorch 實作的 LIF 模型
# --------------------------------------------------------

# ========================================================
# 一、Triangular Surrogate Gradient（三角形替代梯度）
# ========================================================
# 這是論文中明確使用的 surrogate gradient：
# σ′(x) = max(0, 1 - |x|/a) / a
# 只在 |x| < a 範圍內有梯度，其他地方為 0。
# 好處：避免梯度爆炸，且只在閾值附近更新。
# ========================================================

class TriangularSurrogate(torch.autograd.Function):        # 定義自訂 autograd 函式（前向+反向）
    @staticmethod
    def forward(ctx, x, a=1.0):                            # forward: 前向傳遞，ctx 可用來存值
        ctx.save_for_backward(x)                           # 把 x（膜電位減閾值）存起來，反向時用
        ctx.a = a                                          # 把參數 a 存起來（控制梯度範圍）
        return (x >= 0).to(x.dtype)                        # Heaviside step：若 U≥θ 輸出1，否則0
        # ✳️ 這一步對應 SpikingRx 的「S[t] = H(U[t] - θ)」
        # ✳️ forward 回傳的就是這個時間步的 spike（二值 0/1）

    @staticmethod
    def backward(ctx, grad_output):                        # backward: 反向傳遞時呼叫
        (x,) = ctx.saved_tensors                           # 取出 forward 時存的 x
        a = ctx.a                                          # 取出平滑控制參數 a
        grad_input = grad_output.clone()                   # 複製上游梯度 (∂L/∂S)

        # Triangular surrogate 公式：σ′(x) = max(0, 1 - |x|/a) / a
        mask = (x.abs() < a).to(x.dtype)                   # 只在 |x| < a 的範圍內保留梯度
        grad = grad_input * (1 - x.abs() / a) * mask / a   # 計算替代梯度（在閾值附近非零）
        # ✳️ 這裡就是 surrogate gradient 的精髓：
        #    spike 雖然不可微，但這裡給它一個「倒V形」的假梯度讓它能學。
        return grad, None                                  # 回傳對應於 (x, a) 的梯度（a不需要梯度）

# --------------------------------------------------------
# spike_fn 是包裝好的方便函式
# 讓外部直接呼叫 spike_fn(U - θ) 產生 spike。
# --------------------------------------------------------
def spike_fn(x, a=1.0):
    return TriangularSurrogate.apply(x, a)                 # 使用上面定義的自訂 autograd function


# ========================================================
# 二、LIF 模型 (Leaky Integrate-and-Fire Neuron)
# ========================================================
# 在 SpikingRx 的 SEW-ResNet block 中，
# LIF 是「spiking activation」：
#   Conv 輸出連續值特徵 → 經 Norm 正規化 → 傳入 LIF
#   LIF 會隨時間步 t 累積膜電位，超過閾值 θ 時發出 spike。
# ========================================================

class LIF(nn.Module):                                      # 定義一個神經元模組
    """
    符合 SpikingRx 論文的離散時間 LIF 模型：
      U[t] = βU[t−1] + (1−β)I[t]             ← 膜電位積分（leaky）
      S[t] = spike_fn(U[t] − θ)               ← 判斷是否跨閾產生spike
      U[t] = U[t] − S[t]θ                     ← 發泡後 soft reset
    """

    def __init__(self, beta=0.9, theta=1.0, learn_beta=False):
        super().__init__()                    # 初始化父類別
        if learn_beta:
            # 若 learn_beta=True，讓 β 成為可訓練參數（但論文中固定β=0.9）
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            # 否則註冊成 buffer（參數固定，但會隨 model.to(device) 一起移動）
            self.register_buffer("beta", torch.tensor(float(beta)))

        # 閾值 θ：當膜電位超過此值時產生spike
        self.register_buffer("theta", torch.tensor(float(theta)))

    # ----------------------------------------------------
    # forward: LIF 的主要時序運算
    # I: [B, T, C, H, W] （卷積輸出）或 [B, T, D]（全連接輸出）
    # 對應 SpikingRx 論文中：每一個 block 都在 T 個時間步上運作。
    # ----------------------------------------------------
    def forward(self, I):
        if I.dim() == 5:                                     # [B, T, C, H, W]
            B, T, C, H, W = I.shape
            U = I.new_zeros((B, C, H, W))                    # 初始膜電位 U[0]=0
            out_spikes = []                                  # 用來收集每個時間步的 spike
            for t in range(T):                               # 時間展開：模擬 T 個時間步
                It = I[:, t]                                 # 取第 t 個時間步的輸入
                U = self.beta * U + (1 - self.beta) * It     # 更新膜電位（leaky integration）
                # ✳️ 對應論文公式：U[t] = βU[t−1] + (1−β)I[t]

                S = spike_fn(U - self.theta)                 # 判斷是否放電（跨閾值）
                # ✳️ 對應論文：S[t] = H(U[t] − θ)
                # ✳️ forward 時這裡是 0/1，但 backward 時用 triangular surrogate 傳梯度

                U = U - S * self.theta                       # 放電後 soft reset
                # ✳️ 對應論文：U[t] = U[t] − S[t]θ
                # ✳️ 若放電 S=1 → 減掉 θ；若沒放電 S=0 → 保留目前膜電位。

                out_spikes.append(S)                         # 把這個時間步的 spike 存起來

            return torch.stack(out_spikes, dim=1)            # 組成 spike train [B, T, C, H, W]
            # ✳️ SpikingRx：每層 LIF 都會輸出一串 spike train，傳給下一個 block。
            # ✳️ ANN 最後一層會整合所有時間步的 spike 來算 LLR。

        elif I.dim() == 3:                                   # [B, T, D] → 用於最後全連接層
            B, T, D = I.shape
            U = I.new_zeros((B, D))                          # 初始化膜電位
            out_spikes = []
            for t in range(T):
                It = I[:, t]                                 # 第 t 步輸入
                U = self.beta * U + (1 - self.beta) * It     # 更新膜電位
                S = spike_fn(U - self.theta)                 # 是否放電
                U = U - S * self.theta                       # 重設
                out_spikes.append(S)
            return torch.stack(out_spikes, dim=1)            # [B, T, D]

        else:
            raise ValueError("LIF input must be [B,T,C,H,W] or [B,T,D]")
            # ✳️ 確保輸入維度正確：SpikingRx 的輸入一定有時間步 T。

# --------------------------------------------------------
# 模組小結：
# - ArcTanSurrogate 改成 TriangularSurrogate → 與論文一致。
# - LIF 方程與 β, θ 完全對應論文。
# - forward() 的時間展開對應 SpikingRx 中的時序演化。
# - 最終輸出的 spike train 會進入下一個 SEW block 或 ANN 層。
# --------------------------------------------------------
