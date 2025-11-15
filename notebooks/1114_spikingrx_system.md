<img width="1115" height="165" alt="image" src="https://github.com/user-attachments/assets/15f3872f-0eb1-4bbd-b494-d0819d644e75" />

<img width="3427" height="164" alt="Untitled diagram-2025-11-15-071344" src="https://github.com/user-attachments/assets/81acc98a-cd57-47a7-8f44-d477fc7137a5" />

<img width="5047" height="164" alt="Untitled diagram-2025-11-15-071442" src="https://github.com/user-attachments/assets/dc66bbd5-fccd-42c6-9aba-42351f2436ef" />

<img width="3204" height="164" alt="Untitled diagram-2025-11-15-071528" src="https://github.com/user-attachments/assets/5cf21f0d-cffb-4c25-b56f-f536921bf39b" />
# 1115 SpikingRx 系統架構總整理

> 目標：  
> 用「四層流程圖」從**系統級 → 訓練流程 → 模型 forward → 單一 SEW block**，  
> 把 SpikingRx 在 5G PDSCH 接收鏈中的角色、資料流與維度講清楚。

---

## 0. 符號與維度說明（B / T / C / H / W）

後面四層的所有流程圖都會用同一套維度記號：

- **B**：batch size  
  - 一次訓練 / 推論同時處理幾個 OFDM frame  
- **T**：SNN 的時間步數  
  - 同一張 OFDM I/Q 網格，重複丟進網路 **T 次**，讓 LIF 累積電位 → spike train  
- **C**：通道數（channel）  
  - 最一開始 C=2（I / Q），後面經過 Conv / SEW 會變成 16 / 32 / 64 …  
- **H, W**：OFDM 的 2D 網格座標  
  - 例如：  
    - H：OFDM symbol index  
    - W：subcarrier index  

### 0.1 SpikingRx 的核心 I/O

- **整體輸入（Receiver 角度）**  
  - FFT+CP removal 後的 **OFDM I/Q 網格**  
  - `x ∈ ℝ^{B × T × 2 × H × W}`  

- **SpikingRx 輸出給 LDPC decoder 的結果**  
  - 每一個 RE 對應 QPSK 的 2 個 LLR  
  - `LLR ∈ ℝ^{B × H × W × 2}`  

---

## 1. 第 1 層：系統級 Receiver 架構

> 關鍵問題：  
> SpikingRx 在**整個 5G PDSCH 接收鏈**中，取代了哪一段？輸入 / 輸出是什麼？

### 1.1 流程圖（系統級）

> 這裡放你第一張圖的截圖（比如）：  
> `![Layer1_System](images/1115_layer1_system.png)`

右 → 左的大致流程：

- **FFT + CP Removal**  
  - 從 time-domain 信號：
    - 移除 cyclic prefix  
    - 對每個 OFDM symbol 做 FFT  
  - 輸出：頻域的 **OFDM I/Q 資料格**  

- **SpikingRx Receiver (StemConv + 6×SEW + Readout)**  
  - 吃進 `x(B,T,2,H,W)`  
  - 在 SNN 中做：特徵抽取、spiking 時間整合、非線性處理、readout ANN  
  - 最後吐出每個 RE 的 bit-wise LLR  

- **Transport Block Decoding**  
  - 把 SpikingRx 給的 LLR 丟進：
    - Rate dematcher  
    - LDPC decoder  
    - TB 解碼  
  - 從 LLR → bit → TB

### 1.2 第 1 層的資料形狀

- **FFT → SpikingRx：**

  `Input OFDM Grid: x ∈ ℝ^{B×T×2×H×W}`

  - B：一次處理多少個 frame  
  - T：SNN time steps（同一張 OFDM 網格在時間上重複 T 次）  
  - 2：I / Q 兩個通道  
  - H×W：時頻平面 (RE grid)

- **SpikingRx → TB Decoder：**

  `LLR Output: LLR ∈ ℝ^{B×H×W×2}`

  - 對 QPSK，每個 RE 有 2 bits → 2 個 LLR

---

## 2. 第 2 層：訓練一個 Epoch 的流程（Forward / Backward / Update）

> 關鍵問題：  
> 在訓練 SpikingRx 時，**一個 epoch 裡，每個 batch 做了哪些步驟？**

### 2.1 流程圖（Epoch）

> 這裡放你的「Epoch 流程」那張圖：  
> `![Layer2_Epoch](images/1115_layer2_epoch.png)`

右 → 左流程可以整理為：

1. **Training Batch Input**  
   - `x ∈ ℝ^{B×T×2×H×W}`：一批 OFDM I/Q grid（SNN input）  
   - `b ∈ {0,1}^{B×H×W×2}`：對應每個 RE 的真實 bit label  

2. **Forward Pass through SpikingRx**  
   - 用目前的權重 θ，做一次完整 forward：  
     `LLR_hat = SpikingRx(x; θ)`  
   - `LLR_hat ∈ ℝ^{B×H×W×2}`  

3. **Loss Computation**  
   - 以 bit-wise binary cross-entropy 為例：  
     
     \[
     L(\theta) = \frac{1}{BHW2} \sum_{b,h,w,k} \text{BCE}(\text{LLR\_hat}_{b,h,w,k},\, b_{b,h,w,k})
     \]

4. **Backward Pass**  
   - 對 L 對所有參數 θ 求梯度：
     - Readout ANN  
     - SEW6, …, SEW1  
     - StemConv  
   - 使用 surrogate gradient 近似 LIF 的梯度，讓 SNN 可以做反向傳遞。  

5. **Optimizer Update**  
   - 以 Adam 為例：  
     
     \[
     \theta \leftarrow \theta - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
     \]

6. **Epoch 結束**  
   - 所有 batch 都跑完上面流程一次 → 組成一個 epoch。  
   - 紀錄：
     - training loss 曲線  
     - spike rate per stage  
     - LLR 統計  

### 2.2 這一層的重點

- **第 1 層**是「系統在哪裡用 SpikingRx？」  
- **第 2 層**是「怎麼把 SpikingRx 訓練出來？」  
- 這一層只 care：  
  - 進來：`(x, b)`  
  - 出去：更新後的 `θ_new`  
  - 中間：Forward → Loss → Backward → Update 的順序

---

## 3. 第 3 層：SpikingRx Forward Pipeline（模型內部資料流）

> 關鍵問題：  
> 不考慮訓練，只看「推論」，一個 `x(B,T,2,H,W)` 是如何變成 `LLR(B,H,W,2)`？

### 3.1 流程圖（Forward Pipeline）

> 這裡可以直接用你拆成兩段的那 **兩張 forward 圖**：  
> - `![Layer3_Forward_Part1](images/1115_layer3_forward_part1.png)`  
> - `![Layer3_Forward_Part2](images/1115_layer3_forward_part2.png)`

從右到左整理成一條資料流：

1. **OFDM Grid Input**  
   - `x ∈ ℝ^{B×T×2×H×W}`  

2. **StemConv (2→16)**  
   - Conv3×3 + SpikeNorm + LIF  
   - 特徵抽取 + 時間上的 spiking  
   - 輸出：`z₀ ∈ ℝ^{B×T×16×H×W}`

3. **SEW Block 1：16 → 16**  
4. **SEW Block 2：16 → 32**  
5. **SEW Block 3：32 → 32**  
6. **SEW Block 4：32 → 64**  
7. **SEW Block 5：64 → 64**  
8. **SEW Block 6：64 → 64**

   - 每一個 SEW block：  
     - 時間維度 T 保留（對每個 t 做一次 SEW）  
     - channel 依設計改變（16 / 32 / 64）  
   - 一般來說，中間會有類似：  

     - `B,T,16,H,W` → `B,T,16,H,W`  
     - `B,T,16,H,W` → `B,T,32,H,W`  
     - `B,T,32,H,W` → `B,T,32,H,W`  
     - `B,T,32,H,W` → `B,T,64,H,W`  
     - `B,T,64,H,W` → `B,T,64,H,W`（重覆堆疊）

9. **Temporal Mean over T**  
   - 對 spike 輸出在時間 t 上取平均：  

     \[
     r_{b,c,h,w} = \frac{1}{T} \sum_{t} y_{b,t,c,h,w}
     \]

   - 得到 spike rate map：  
     `r ∈ ℝ^{B×64×H×W}`  

10. **Readout ANN (64→32→2)**  
    - Conv1×1: 64 → 32 → ReLU → Conv1×1: 32 → 2  
    - 對每個 RE 的 64 維特徵做 bit-level logit 推論  
    - 輸出：`logits ∈ ℝ^{B×2×H×W}`  
    - 重排為：`LLR ∈ ℝ^{B×H×W×2}`  

### 3.2 這一層的重點

- 第 3 層只看 **「推論時資料怎麼流」**，不看梯度。  
- 把第 1 層的「SpikingRx Receiver」整個展開成：  
  `x → StemConv → 6×SEW → Time-mean → Readout → LLR`  
- 這一層很適合放在 PPT / 報告裡講「模型結構」。

---

## 4. 第 4 層：單一 SEW Block 的內部結構

> 關鍵問題：  
> 一個 SEW block 裡，「Conv / Norm / LIF / Shortcut」在數學上怎麼串？

### 4.1 流程圖（SEW block）

> 這裡放你第四張圖：  
> `![Layer4_SEW_Block](images/1115_layer4_sew_block.png)`

以**單一時間步 t** 為例，右 → 左流程：

1. **Input：`x_t ∈ ℝ^{B×C_in×H×W}`**  
   - 來自上一層 / 上一個 block 的 feature map，  
   - 對應時間 t 的切片。

2. **Conv2D（主支路 main path）**  
   - kernel：例如 3×3  
   - channel：`C_in → C_out`  
   - 得到：`z_t ∈ ℝ^{B×C_out×H×W}`  

3. **SpikeNorm**  
   - channel-wise 的 normalization  
   - 控制 spiking 的活性範圍，避免全部 0 或全部 1。  

4. **LIF Neuron**  
   - 每個 `(b,c,h,w)` 有一個膜電位 `v`：  
     - 積分：`v(t) = β v(t-1) + z_t`  
     - 產生 spike：`s_t = 1[v(t) > θ]`  
     - reset：`v(t) = v(t) - θ s_t`  

   - 輸出是 spike map `s_t ∈ {0,1}^{B×C_out×H×W}` 或經 clamp 的 [0,1]。

5. **Shortcut 分支**  
   - 如果 `C_in == C_out`：  
     - shortcut 就是 identity：`sc_t = x_t`  
   - 如果 `C_in != C_out`：  
     - 用 1×1 Conv 對 channel 對齊：  
       `sc_t = Conv1×1(x_t)`  

6. **Add（main + shortcut）**  
   - 最終輸出：  
     \[
     y_t = s_t + sc_t
     \]
   - `y_t ∈ ℝ^{B×C_out×H×W}`  

### 4.2 時間軸上的 SEW Block

整個 SEW block 對 **全部時間步** 的行為：

- Input：`X ∈ ℝ^{B×T×C_in×H×W}`  
- Output：`Y ∈ ℝ^{B×T×C_out×H×W}`  
- 實作上通常是：  
  - 在 `forward` 裡面 `for t in range(T)`：  
    - 取出 `x[:,t,...]`  
    - 跑一遍 `Conv → Norm → LIF → Add`  
    - 堆回一個 `out_seq` list  
  - 最後用 `torch.stack(out_seq, dim=1)` 還原成 `B,T,C_out,H,W`。

---

## 5. 建議閱讀順序（給看你 GitHub 的人）

1. **先看第 1 層**：知道 SpikingRx 在整個 5G 收發鏈裡的位置。  
2. **再看第 2 層**：懂得訓練時一個 epoch 的流程。  
3. **接著看第 3 層**：清楚 SpikingRx 裡資料怎麼從 `x` 流到 `LLR`。  
4. **最後看第 4 層**：了解 SEW block 內部細節與 LIF 的角色。  

這樣從「系統 → 訓練流程 → 模型 → block 細節」的順序，  
可以很有系統地讀你整個 SpikingRx 專案。

---
