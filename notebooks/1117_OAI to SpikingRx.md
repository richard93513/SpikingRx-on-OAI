# ✅ 2025/11/17 --- OAI → SpikingRx 全流程完整技術報告

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

# 0. 系統整體架構流程概述

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

本報告記錄從 **OAI gNB/UE 建置 → PDSCH dump → Python 32×32 映射 →
SpikingRx 輸入生成** 的完整技術流程。\
包含：

1.  系統環境

2.  OAI 編譯

3.  rfsim 連線

4.  UE PDSCH dump（C 程式）

5.  600 RE → 32×32 映射（Python）

6.  tensor 驗證（Python）

7.  三段程式的完整技術分析

8.  圖片（由使用者自行貼入）

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

# 1. 系統環境與 OAI 初始化

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

## 1.1 Ubuntu 版本

-   Ubuntu 22.04.6 LTS（與 OAI 相容性最佳）

## 1.2 OAI 依賴安裝

sudo apt update

sudo apt install git build-essential cmake ninja-build \\

python3 python3-pip python3-venv \\

libboost-all-dev libsctp-dev libconfig++-dev

## 1.3 下載 OAI

cd \~

git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git

cd openairinterface5g

## 1.4 編譯 OAI（gNB + nrUE）

### 清 Cache：

cd \~/openairinterface5g/cmake_targets/ran_build/build

rm -rf CMakeCache.txt CMakeFiles

### 正式編譯：

cd \~/openairinterface5g/cmake_targets

sudo ./build_oai \--gNB \--nrUE -w SIMU

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

# 2. rfsim 啟動與連線

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

## 2.1 啟動 gNB

cd \~/openairinterface5g/cmake_targets/ran_build/build

sudo ./nr-softmodem \--rfsim \--sa \\

-O ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.conf

## 2.2 啟動 UE（新 terminal）

cd \~/openairinterface5g/cmake_targets/ran_build/build

sudo ./nr-uesoftmodem \--rfsim \--sa

## 2.3 連線成功畫面

![](media/image1.png){width="6.036458880139983in"
height="8.391162510936134in"}

建議顯示：

-   RSRP / SNR

-   HARQ

-   RNTI 建立

-   slot 遞增

-   UE 已成功解調 PDSCH

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

# 3. 完整 dump 流程：rxdataF_ext（slot-level）

# =============================================

# A. PDSCH dump（UE端 C 程式）

# =============================================

# A.1 程式碼

if (scope_req-\>copy_rxdataF_to_scope) {

size_t size = sizeof(c16_t) \* nb_re_pdsch;

int copy_index = symbol - dlsch_config-\>start_symbol;

UEscopeCopyUnsafe(ue, pdschRxdataF, rxdataF_ext\[0\], size,
scope_req-\>scope_rxdataF_offset, copy_index);

scope_req-\>scope_rxdataF_offset += size;

}

// ===== SpikingRx: slot-level PDSCH RX I/Q dump =====

// 條件：只在第 0 個 codeword、第 0 根 RX 天線上做，避免重複 dump 太多

if (nb_re_pdsch \> 0 && nbRx \> 0) {

// 如果還沒開始 dump，且這個 symbol 是 PDSCH 的起始 symbol，就初始化本
slot 的檔案

if (!spx_dump_active && symbol == dlsch_config-\>start_symbol) {

spx_dump_frame = frame;

spx_dump_slot = nr_slot_rx;

spx_dump_start_symbol = symbol;

spx_dump_active = 1;

// 先清空舊檔案

FILE \*f = fopen(\"/tmp/ue_pdsch_slot_rxdataF_ext.bin\", \"wb\");

if (f != NULL) {

fclose(f);

LOG_I(PHY,

\"\[SpikingRx\] Start new PDSCH dump: frame %d slot %d start_symbol %d
nb_rb=%d\\n\",

frame,

nr_slot_rx,

dlsch_config-\>start_symbol,

nb_rb_pdsch);

} else {

LOG_E(PHY, \"\[SpikingRx\] Failed to create
/tmp/ue_pdsch_slot_rxdataF_ext.bin\\n\");

spx_dump_active = 0;

}

// 同時輸出一個簡單的 meta 資訊檔案

FILE \*meta = fopen(\"/tmp/ue_pdsch_slot_meta.txt\", \"w\");

if (meta != NULL) {

fprintf(meta,

\"frame %d\\nslot %d\\nstart_symbol %d\\nnumber_symbols %d\\nnb_rb_pdsch
%d\\n\",

frame,

nr_slot_rx,

dlsch_config-\>start_symbol,

dlsch_config-\>number_symbols,

nb_rb_pdsch);

fclose(meta);

}

}

// 若這個 symbol 屬於同一個 slot 的 PDSCH，就把這個 symbol 的
rxdataF_ext\[0\] 接在檔案後面

if (spx_dump_active &&

frame == spx_dump_frame &&

nr_slot_rx == spx_dump_slot &&

symbol \>= spx_dump_start_symbol &&

symbol \< dlsch_config-\>start_symbol + dlsch_config-\>number_symbols) {

FILE \*f = fopen(\"/tmp/ue_pdsch_slot_rxdataF_ext.bin\", \"ab\");

if (f != NULL) {

size_t written = fwrite(rxdataF_ext\[0\], sizeof(c16_t), nb_re_pdsch,
f);

fclose(f);

LOG_I(PHY,

\"\[SpikingRx\] Appended PDSCH symbol %d: nb_re_pdsch=%u,
written=%zu\\n\",

symbol,

nb_re_pdsch,

written);

} else {

LOG_E(PHY, \"\[SpikingRx\] Failed to append to
/tmp/ue_pdsch_slot_rxdataF_ext.bin\\n\");

spx_dump_active = 0;

}

}

// 如果這個 symbol 已經是本 PDSCH 的最後一個 symbol，就關閉本次 dump

if (spx_dump_active &&

symbol == dlsch_config-\>start_symbol +
dlsch_config-\>number_symbols - 1) {

LOG_I(PHY,

\"\[SpikingRx\] Finished PDSCH dump for frame %d slot %d (symbols
%d..%d)\\n\",

spx_dump_frame,

spx_dump_slot,

spx_dump_start_symbol,

dlsch_config-\>start_symbol + dlsch_config-\>number_symbols - 1);

spx_dump_active = 0;

}

}

// ===== end of SpikingRx slot-level dump =====

# A.2 程式碼技術解釋（逐段解析）

以下針對 dump 程式分為四大邏輯段落，完全對應程式區塊：

## 1️⃣ 判斷是否為有效 PDSCH RE

條件：

-   nb_re_pdsch \> 0 → 此 symbol 內存在 PDSCH RE

-   nbRx \> 0 → UE 至少有一根接收天線

此段負責確保：

-   只在真正包含 PDSCH 的 symbol 上觸發 dump 邏輯

-   避免空 symbol 或 DMRS-only symbol 造成誤判

## 2️⃣ 若 symbol 為 PDSCH 起始 symbol → 初始化整個 slot 的 dump

動作內容：

1.  記錄此 slot 的 frame、slot、start_symbol

2.  開啟 ascii meta 檔案並寫入：

    -   frame

    -   slot

    -   start_symbol

    -   number_symbols

    -   PDSCH 使用的 RB 數

3.  使用 \"wb\" 方式清空舊的 .bin dump 檔案

4.  啟動 dump flag (spx_dump_active = 1)

## 3️⃣ 若 symbol 屬於此次的 PDSCH → append 寫入 rxdataF_ext

資料按 symbol 依序排列：

Symbol0 → 60 RE

Symbol1 → 60 RE

\...

Symbol9 → 60 RE

共 600 RE

## 4️⃣ 若 symbol 為 PDSCH 最後一個 symbol → 結束 dump

-   停止此次 dump

-   印出結束訊息

# A.3 Terminal 執行指令

gNB:

sudo ./nr-softmodem \--rfsim \--sa \...

UE:

sudo ./nr-uesoftmodem \--rfsim \--sa

# A.4 dump 結果圖

![](media/image4.png){width="9.572916666666666in"
height="10.041666666666666in"}

# A.5 dump 結果圖解釋

圖中一般可見：

-   正常的 RSRP/SNR

-   RNTI 建立

-   HARQ

-   \[SpikingRx\] Start new PDSCH dump

-   \[SpikingRx\] Appended symbol \...

-   \[SpikingRx\] Finished \...

# =============================================

# B. OAI → 32×32 映射程式（Python）

# =============================================

# B.1 程式碼

import numpy as np

import torch

def oai_to_spikingrx_grid(path, H=32, W=32):

raw = np.fromfile(path, dtype=np.int16).reshape(-1, 2)

SYM = 10

SC = 60

assert raw.shape\[0\] == SYM \* SC

cpx = raw\[:,0\] + 1j \* raw\[:,1\]

grid = cpx.reshape(SYM, SC)

\# Step 1: freq compress 60 -\> 32

new_SC = W

ratio = SC / new_SC

compressed = np.zeros((SYM, new_SC), dtype=np.complex64)

for i in range(new_SC):

s = int(i \* ratio)

e = int((i+1) \* ratio)

if e == s: e = s+1

compressed\[:, i\] = grid\[:, s:e\].mean(axis=1)

\# Step 2: pad time 10-\>32

out = np.zeros((H, W), dtype=np.complex64)

out\[:SYM, :\] = compressed

\# Step 3: build tensor C,H,W

tensor = np.zeros((2, H, W), dtype=np.float32)

tensor\[0\] = out.real

tensor\[1\] = out.imag

return torch.tensor(tensor)

# B.2 程式碼技術解釋（逐段解析）

## 1️⃣ 載入 OAI dump

-   每筆資料為 (I, Q) → int16

-   reshape 為 (600, 2)

## 2️⃣ 轉成複數並 reshape 成 (symbol, subcarrier)

-   (600,) → (10,60)

## 3️⃣ 頻域子載波壓縮：60 → 32

-   每段均值壓縮

-   壓縮後形狀 (10,32)

## 4️⃣ 時間維度 padding：10 → 32

形成：

(32, 32)

## 5️⃣ 拆成 I/Q → (2,32,32)

符合 SpikingRx 輸入介面。

# B.3 Terminal 執行

python3 oai_to_32x32.py

# B.4 結果圖

![](media/image3.png){width="9.59375in" height="5.541666666666667in"}

# B.5 結果圖解釋

-   上方 10 列 → 有效 OFDM symbols

-   下方全 0 → padding

-   水平為壓縮後 32 subcarriers

-   I channel 與 Q channel 分離正常

# ============================================

# C. test.py 驗證程式（Python）

# ============================================

# C.1 程式碼

from oai_to_32x32 import oai_to_spikingrx_grid

path = \"/tmp/ue_pdsch_slot_rxdataF_ext.bin\"

tensor = oai_to_spikingrx_grid(path)

print(\"shape:\", tensor.shape)

print(tensor\[:, :8, :8\]) \# 印前 8x8，會比剛剛更多資訊

# C.2 程式碼技術解釋

## 1️⃣ 載入 32×32 tensor

應輸出：

torch.Size(\[2,32,32\])

## 2️⃣ 印出前 8×8 區塊

-   用來檢查資料完整性

-   驗證 reshape、壓縮、padding 是否成功

# C.3 Terminal 執行

python3 test.py

# C.4 結果圖

![](media/image2.png){width="9.572916666666666in"
height="7.416666666666667in"}

# C.5 結果圖解釋

-   維度正確：\[2,32,32\]

-   前 8×8 有合理振幅

-   I/Q 兩 channel 分離成功

-   下半部 padding 正確為零

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

# 4. 流程總結

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

## 4.1 OAI 端

1.  gNB/UE rfsim 啟動

2.  UE demodulation 擷取 rxdataF_ext

3.  按 symbol append → 600 RE

4.  meta + binary 記錄 slot

## 4.2 Python 端

1.  600 RE 載入

2.  reshape → (10,60)

3.  壓縮 → (10,32)

4.  padding → (32,32)

5.  拆成 I/Q → (2,32,32)

## 4.3 SpikingRx 端

最終輸入格式與模型需求一致：

\[B, T, 2, 32, 32\]

可直接進行：

-   SEW-ResNet 處理

-   LIF spike 展開

-   LLR 預測

-   與 OAI soft-demapper 進行 LLR 對照

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

# 5. 後續步驟：SpikingRx 模型移植、LLR 驗證、OAI 端對接

# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

本章節說明完成 OAI 端 PDSCH dump 與 600RE → 32×32
前處理後，下一階段的整合工作流程。\
此流程描述如何將 Windows 環境下已建立的 SpikingRx 模型移至 Ubuntu，並在
OAI dump 上執行推論、比對 LLR 輸出，最終完成 OAI gNB/UE 與外部 SpikingRx
的串接。
