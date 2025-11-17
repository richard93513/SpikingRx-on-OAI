# 2025/11/17 — Full Technical Report: OAI → SpikingRx Workflow (Final Integrated Version)

## 0. System Overview

This report documents the complete technical pipeline from building OAI gNB/UE → dumping PDSCH → Python 32×32 mapping → generating the input for SpikingRx.

Contents include:

- System environment  
- OAI compilation  
- rfsim connection  
- UE PDSCH dump (C code)  
- 600-RE → 32×32 mapping (Python)  
- Tensor verification (Python)  
- Technical analysis of all three code segments  
- (Images omitted per user request)


---

## 1. System Environment and OAI Initialization

### 1.1 Ubuntu Version  
Ubuntu 22.04.6 LTS (best compatibility with OAI)

### 1.2 Install OAI Dependencies

sudo apt update  
sudo apt install git build-essential cmake ninja-build \
    python3 python3-pip python3-venv \
    libboost-all-dev libsctp-dev libconfig++-dev

### 1.3 Download OAI

cd ~  
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git  
cd openairinterface5g

### 1.4 Compile OAI (gNB + nrUE)

Clear cache:

cd ~/openairinterface5g/cmake_targets/ran_build/build  
rm -rf CMakeCache.txt CMakeFiles

Compile:

cd ~/openairinterface5g/cmake_targets  
sudo ./build_oai --gNB --nrUE -w SIMU


---

## 2. rfsim Startup and Connection

### 2.1 Start gNB

cd ~/openairinterface5g/cmake_targets/ran_build/build  
sudo ./nr-softmodem --rfsim --sa \
  -O ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.conf

### 2.2 Start UE (new terminal)

cd ~/openairinterface5g/cmake_targets/ran_build/build  
sudo ./nr-uesoftmodem --rfsim --sa

### 2.3 Expected Successful Connection Output

You should see:

- RSRP / SNR  
- HARQ  
- RNTI established  
- Slot incrementing  
- UE decoding PDSCH successfully


---

## 3. Full Dump Workflow: rxdataF_ext (Slot-level)

# A. PDSCH Dump (UE-side C Program)

## A.1 Source Code

## A.1 Source Code

if (scope_req->copy_rxdataF_to_scope) {
  size_t size = sizeof(c16_t) * nb_re_pdsch;
  int copy_index = symbol - dlsch_config->start_symbol;
  UEscopeCopyUnsafe(ue, pdschRxdataF, rxdataF_ext[0], size, scope_req->scope_rxdataF_offset, copy_index);
  scope_req->scope_rxdataF_offset += size;
}

// ===== SpikingRx: slot-level PDSCH RX I/Q dump =====
// Only dump on codeword 0 and RX antenna 0 to avoid redundancy

if (nb_re_pdsch > 0 && nbRx > 0) {

  // Start of PDSCH for this slot
  if (!spx_dump_active && symbol == dlsch_config->start_symbol) {
    spx_dump_frame = frame;
    spx_dump_slot = nr_slot_rx;
    spx_dump_start_symbol = symbol;
    spx_dump_active = 1;

    // Clear old dump file
    FILE *f = fopen("/tmp/ue_pdsch_slot_rxdataF_ext.bin", "wb");
    if (f != NULL) {
      fclose(f);
      LOG_I(PHY,
            "[SpikingRx] Start new PDSCH dump: frame %d slot %d start_symbol %d nb_rb=%d\n",
            frame,
            nr_slot_rx,
            dlsch_config->start_symbol,
            nb_rb_pdsch);
    } else {
      LOG_E(PHY, "[SpikingRx] Failed to create /tmp/ue_pdsch_slot_rxdataF_ext.bin\n");
      spx_dump_active = 0;
    }

    // Create metadata file
    FILE *meta = fopen("/tmp/ue_pdsch_slot_meta.txt", "w");
    if (meta != NULL) {
      fprintf(meta,
              "frame %d\nslot %d\nstart_symbol %d\nnumber_symbols %d\nnb_rb_pdsch %d\n",
              frame,
              nr_slot_rx,
              dlsch_config->start_symbol,
              dlsch_config->number_symbols,
              nb_rb_pdsch);
      fclose(meta);
    }
  }

  // Append symbol data
  if (spx_dump_active &&
      frame == spx_dump_frame &&
      nr_slot_rx == spx_dump_slot &&
      symbol >= spx_dump_start_symbol &&
      symbol < dlsch_config->start_symbol + dlsch_config->number_symbols) {

    FILE *f = fopen("/tmp/ue_pdsch_slot_rxdataF_ext.bin", "ab");
    if (f != NULL) {
      size_t written = fwrite(rxdataF_ext[0], sizeof(c16_t), nb_re_pdsch, f);
      fclose(f);
      LOG_I(PHY,
            "[SpikingRx] Appended PDSCH symbol %d: nb_re_pdsch=%u, written=%zu\n",
            symbol,
            nb_re_pdsch,
            written);
    } else {
      LOG_E(PHY, "[SpikingRx] Failed to append to /tmp/ue_pdsch_slot_rxdataF_ext.bin\n");
      spx_dump_active = 0;
    }
  }

  // End of PDSCH
  if (spx_dump_active &&
      symbol == dlsch_config->start_symbol + dlsch_config->number_symbols - 1) {
    LOG_I(PHY,
          "[SpikingRx] Finished PDSCH dump for frame %d slot %d (symbols %d..%d)\n",
          spx_dump_frame,
          spx_dump_slot,
          spx_dump_start_symbol,
          dlsch_config->start_symbol + dlsch_config->number_symbols - 1);
    spx_dump_active = 0;
  }
}

// ===== end of SpikingRx slot-level dump =====


---

## A.2 Technical Explanation of the Code

### 1. Valid PDSCH RE Check

Conditions:

- nb_re_pdsch > 0  → Symbol contains PDSCH RE  
- nbRx > 0         → At least one RX antenna  

Purpose:

- Only run dump during effective PDSCH processing  
- Avoid dumping empty or DMRS-only symbols  


### 2. Start Dump at the First PDSCH Symbol

Actions:

- Record the frame, slot, and start symbol  
- Create metadata file (frame / slot / start_symbol / num_symbols / RB count)  
- Clear previous dump file  
- Activate dump flag  


### 3. Append Each Symbol’s RE Block

Order written into file:

Symbol 0  → 60 RE  
Symbol 1  → 60 RE  
...  
Symbol 9  → 60 RE  

Total: 600 RE appended sequentially.  


### 4. Final Symbol → Close Dump

- Print end message  
- Reset dump flag  


---

## A.3 Terminal Commands

Start gNB:

sudo ./nr-softmodem --rfsim --sa ...

Start UE:

sudo ./nr-uesoftmodem --rfsim --sa


---

## A.4 Dump Result Images

(omitted)


## A.5 Explanation of Dump Results

You should see logs such as:

- RSRP/SNR values    
- RNTI establishment  
- HARQ processes running  
- [SpikingRx] Start new PDSCH dump  
- [SpikingRx] Appended symbol …  
- [SpikingRx] Finished …  


---

# B. OAI → 32×32 Mapping Code (Python)

## B.1 Python Code

import numpy as np
import torch

def oai_to_spikingrx_grid(path, H=32, W=32):
    raw = np.fromfile(path, dtype=np.int16).reshape(-1, 2)
    SYM = 10
    SC = 60
    assert raw.shape[0] == SYM * SC

    cpx = raw[:,0] + 1j * raw[:,1]
    grid = cpx.reshape(SYM, SC)

    # Step 1: compress 60 → 32 subcarriers
    new_SC = W
    ratio = SC / new_SC
    compressed = np.zeros((SYM, new_SC), dtype=np.complex64)
    for i in range(new_SC):
        s = int(i * ratio)
        e = int((i + 1) * ratio)
        if e == s:
            e = s + 1
        compressed[:, i] = grid[:, s:e].mean(axis=1)

    # Step 2: pad 10 → 32 symbols
    out = np.zeros((H, W), dtype=np.complex64)
    out[:SYM, :] = compressed

    # Step 3: build tensor (C, H, W) with I/Q
    tensor = np.zeros((2, H, W), dtype=np.float32)
    tensor[0] = out.real
    tensor[1] = out.imag
    return torch.tensor(tensor)


---

## B.2 Technical Explanation (Step-by-step)

1) Load OAI dump

- Use np.fromfile to read the binary file at "path"  
- dtype=np.int16, because each I/Q component from OAI is 16-bit signed integer  
- reshape(-1, 2) → each row = [I, Q]  

Result shape:

- raw.shape = (600, 2)  
- 600 = 10 symbols × 60 subcarriers  


2) Convert to complex and reshape to (SYM, SC)

- cpx = raw[:,0] + 1j * raw[:,1]  
  - Real part = I  
  - Imag part = Q  
- grid = cpx.reshape(SYM, SC)  
  - SYM = 10 (OFDM symbols)  
  - SC  = 60 (subcarriers used by PDSCH in this configuration)  

Result: grid shape = (10, 60)  


3) Frequency compression: 60 → 32

Goal:

- Map 60 subcarriers into 32 "compressed" subcarriers  
- Use average pooling over frequency intervals  

Details:

- new_SC = W = 32  
- ratio = SC / new_SC = 60 / 32  
- For each i in [0, 31]:

  - s = int(i * ratio)  
  - e = int((i + 1) * ratio)  
  - If e == s, force e = s + 1 (avoid empty segment)  
  - compressed[:, i] = mean over grid[:, s:e] (i.e., average adjacent subcarriers)  

Result: compressed shape = (10, 32)  


4) Time padding: 10 → 32 symbols

Goal:

- SpikingRx expects a 32×32 grid  
- OAI only gives 10 PDSCH symbols in this slot  

Method:

- out = zeros((H, W)) = zeros((32, 32))  
- out[:SYM, :] = compressed  
  - First 10 rows = real data  
  - Remaining 22 rows = zeros  

Result: out shape = (32, 32)  


5) Build final tensor (C, H, W)

- tensor = zeros((2, H, W))  
- tensor[0] = out.real (I-channel)  
- tensor[1] = out.imag (Q-channel)  

Return:

- torch.tensor(tensor)  
- Final shape: (2, 32, 32)  

This matches the SpikingRx model input format for a single example:

- C = 2 (I/Q)  
- H = 32 (time / symbol index)  
- W = 32 (frequency / subcarrier index)  


---

## B.3 Terminal Command to Run the Mapping

Example usage in a standalone script (e.g., oai_to_32x32.py):

- Make sure the file contains the function "oai_to_spikingrx_grid"  
- Then execute:

python3 oai_to_32x32.py

(Or directly import and call the function inside another Python script.)


---

## B.4 Result Images

(omitted)


## B.5 Explanation of Visualization

If you visualize the resulting tensor (for example, plotting the I-channel as a heatmap):

- Top 10 rows: actual OFDM symbols containing PDSCH data  
- Bottom 22 rows: zero padding (no data)  
- Horizontal axis (32 columns): compressed subcarriers after mean pooling  
- Channel 0: real part (I)  
- Channel 1: imaginary part (Q)  


---

# C. test.py Verification (Python)

## C.1 Code

from oai_to_32x32 import oai_to_spikingrx_grid

path = "/tmp/ue_pdsch_slot_rxdataF_ext.bin"
tensor = oai_to_spikingrx_grid(path)

print("shape:", tensor.shape)
print(tensor[:, :8, :8])  # print the first 8x8 block for inspection


---

## C.2 Explanation

1) Load 32×32 tensor

- Call oai_to_spikingrx_grid(path)  
- Expected output:

shape: torch.Size([2, 32, 32])

2) Print the first 8×8 region

- tensor[:, :8, :8] prints:

  - Channel dimension: 2 (I/Q)  
  - First 8 rows (time) and 8 columns (frequency)  

Purpose:

- Check that the numbers are reasonable (not all zeros, not NaN)  
- Confirm that reshaping, compression, and padding are functioning correctly  


---

## C.3 Terminal Command

python3 test.py


---

## C.4 Result Images

(omitted)


## C.5 Explanation of test.py Output

From a correct run, you should see:

- "shape: torch.Size([2, 32, 32])"  
- A 2×8×8 block of floating-point values (I/Q)  
- Values in the top-left region corresponding to the first few symbols and subcarriers  
- Zero values in padded areas (if you print deeper into the 32×32 grid)  


---

## 4. Workflow Summary

### 4.1 OAI Side

- Launch gNB/UE with rfsim  
- UE demodulator extracts rxdataF_ext per symbol  
- For each symbol within the PDSCH duration:

  - Append 60 RE to the binary dump file  
  - Total 10 symbols → 600 RE  

- At the same time, write metadata including:

  - frame  
  - slot  
  - start_symbol  
  - number_symbols  
  - nb_rb_pdsch  


### 4.2 Python Side

- Load 600 RE from "/tmp/ue_pdsch_slot_rxdataF_ext.bin"  
- Reshape:

  - 600 → (10, 60)  

- Compress in frequency:

  - (10, 60) → (10, 32)  

- Pad in time:

  - (10, 32) → (32, 32)  

- Build final tensor:

  - Split real/imag → (2, 32, 32)  


### 4.3 SpikingRx Side

The final input format for a single example becomes:

[ C, H, W ] = [ 2, 32, 32 ]

When batching for SpikingRx, it can be extended to:

[ B, T, C, H, W ]

Where:

- B = batch size (number of frames processed in parallel)  
- T = SNN time steps (repeats of the same (C, H, W) input over time)  
- C = 2 (I/Q)  
- H = 32 (OFDM symbol index)  
- W = 32 (subcarrier index)  

This tensor can be fed into:

- SEW-ResNet backbone (SpikingRx architecture)  
- LIF neuron layers to generate spike trains  
- Final fully connected / readout layers to predict LLRs for each bit  

The predicted LLRs can then be directly compared with:

- LLRs from the traditional OAI soft demapper  
- Downstream LDPC decoder performance (BER/BLER)  


---

## 5. Next Steps: SpikingRx Model Migration, LLR Validation, OAI Integration

After completing:

- PDSCH dump on the UE side  
- 600 RE → 32×32 tensor preprocessing on the Python side  

the next integration steps are:

1) Move the trained SpikingRx model (currently on Windows) to the Ubuntu environment running OAI.  

2) Load the model in Python on Ubuntu and feed the tensors generated from actual OAI dumps.  

3) Compute LLRs using SpikingRx and compare them with:

   - OAI soft-demapper LLRs  
   - Baseline NeuralRx (if available)  

4) Gradually integrate the SpikingRx inference into the OAI workflow, for example:

   - As an external script that reads dumps and outputs LLRs  
   - Or as a future deeper integration point in the OAI receiver chain  

5) Evaluate end-to-end performance:

   - Bit Error Rate (BER)  
   - Block Error Rate (BLER)  
   - Robustness under different SNR / channel models  
   - Latency and complexity compared to the traditional receiver modules  

This completes the full pipeline from:

OAI gNB/UE (rfsim) → PDSCH dump → 600 RE → 32×32 mapping → SpikingRx input format preparation → next-step integration for LLR verification and 5G receiver replacement experiments.
