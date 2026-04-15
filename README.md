# SpikingRx-on-OAI

SpikingRx-on-OAI integrates a spiking neural network-based receiver into the OpenAirInterface (OAI) 5G NR downlink PHY.

The system performs end-to-end inference on FFT-domain PDSCH resource grids and produces log-likelihood ratios (LLRs), which are evaluated through the standard OAI rate recovery and LDPC decoding chain.

---

# 1. System Overview

This project implements a **receiver-level replacement study** in a controlled 5G NR simulation environment.

We compare two receivers under strictly identical conditions:

### (1) Baseline Receiver (OAI)

- Channel estimation (LS + interpolation)  
- Equalization (LMMSE)  
- Soft demapping (LLR generation)  

### (2) Proposed Receiver (SpikingRx)

- Spiking neural network (SEW-ResNet with LIF dynamics)  
- Direct prediction of:
  - Channel estimate  
  - Equalized symbols  
  - Log-likelihood ratios (LLRs)  

---

## System Architecture

<img src="https://github.com/user-attachments/assets/fd7a7891-5aa2-4c30-9feb-3e19c5dfbb9c" width="100%">

---

# 2. System Model

We consider a SISO OFDM-based 5G NR downlink system:

```
Y[k] = H[k] X[k] + N[k]
```

where:

- \(X[k]\): transmitted symbol  
- \(H[k]\): channel response  
- \(N[k]\): AWGN  
- \(Y[k]\): received signal  

The objective is to recover the transmitted transport block bits from \(Y[k]\).

---

# 3. Experimental Fairness Principle

To ensure a valid system-level comparison:

- Identical FFT-domain input grids  
- Identical channel realizations  
- Identical noise conditions  
- Identical LDPC decoding backend  

Only the **receiver frontend is replaced**.

---

# 4. Full Pipeline

```text
┌────────────────────────────┐
│ OAI gNB / UE (rfsim mode)  │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ fullgrid.bin               │
│ dmrs_mask.bin              │
│ data_mask.bin              │
│ demapper_llr_f32.bin       │
│ txbits.bin                 │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ oai_to_spikingrx_tensor.py │
│ - circular reorder         │
│ - tensor reshape           │
│ - output [1,1,4,14,1272]   │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ SpikingRxModel             │
│ ch → eq → llr              │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ inferred_llr.bin           │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ rmunmatch_spx              │
│ ldpctest_spx               │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ decoded_bits.bin           │
│ BER vs txbits.bin          │
└────────────────────────────┘
```

---

# 5. Dataset Format

Each bundle contains:

```
fullgrid.bin
dmrs_mask.bin
data_mask.bin
demapper_llr_f32.bin
txbits.bin
ldpc_cfg.txt
pdsch_cfg.txt
```

---

# 6. Carrier Reordering (Critical)

```python
idx = (np.arange(used_sc) + first_carrier_offset) % n_sc_full
```

Without this:

- DMRS misalignment  
- incorrect supervision  
- invalid BER  

---

# 7. Input Representation

```
[B, T, C, H, W] = [B, 1, 4, 14, 1272]
```

---

# 8. Model Architecture

```
shared encoder
→ channel head
→ equalization head
→ LLR head
```

---

# 9. LLR Calibration

```
LLR' = LLR / T   (T = 2.0)
```

---

# 10. Training

```bash
python src/train/train_spikingrx_oai.py \
    --dataset ./dataset_oai_bundle \
    --epochs 100 \
    --batch-size 8
```

---

# 11. Inference

```bash
python src/inference/infer_oai_serial.py \
    --bundle example_bundle/sample_0001 \
    --ckpt checkpoints/spikingrx_oai_serial_best.pt \
    --out inferred_llr.bin
```

---

# 12. BER Evaluation

```bash
python src/inference/check_oai_llr_decode.py \
    inferred_llr.bin \
    ldpc_cfg.txt \
    pdsch_cfg.txt \
    txbits.bin \
    --rmunmatch ./oai_change/rmunmatch_spx \
    --ldpctest ./oai_change/ldpctest_spx
```

---

# 13. Metric Definition

```
BER = (# erroneous bits) / (total transmitted bits)
```

- post-LDPC  
- averaged across bundles  

---

# 14. Experimental Results

Stored under:

```
spx_records/snapshots_snr/
```

Includes:

- BER vs SNR  
- SpikingRx vs OAI comparison  
- LLR statistics  
- case-wise analysis  

---

# 15. Limitations

- QPSK only  
- single code block  
- single slot  
- AWGN channel  

---

# 16. Future Work

- higher modulation  
- MIMO  
- BLER  
- hardware deployment  

---

# 17. Repository Layout

```
SpikingRx-on-OAI/
├── src/
├── oai_change/
├── spx_records/
├── checkpoints/
├── example_bundle/
├── legacy/
```

---

# 18. Dataset

```
https://drive.google.com/file/d/1vO04jncqe-hFHiepl01yRuGgezznBQ16/view
```

---

# 19. Notes

- Raw OAI dumps are not included due to storage constraints  
- Only processed results are provided  
- Fully reproducible from bundle format  

---
### ================================
### END
### ================================
