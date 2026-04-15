# SpikingRx-on-OAI

SpikingRx-on-OAI integrates a spiking neural network-based receiver into the OpenAirInterface (OAI) 5G NR downlink physical layer.

The system performs end-to-end learning on full FFT-domain PDSCH resource grids and outputs log-likelihood ratios (LLRs), which are evaluated through the standard OAI rate recovery and LDPC decoding chain.

---

# 1. System Overview

This project implements a **receiver-level replacement study** in a controlled 5G NR simulation environment.

We compare two receivers under strictly identical conditions:

### (1) Baseline Receiver (OAI)
- Channel estimation (LS + interpolation)
- Equalization (LMMSE)
- Soft demapping (LLR generation)

### (2) Proposed Receiver (SpikingRx)
- Spiking neural network (SEW-ResNet + LIF dynamics)
- Direct prediction of:
  - Channel estimate
  - Equalized symbols
  - Log-likelihood ratios (LLRs)

---

## System Architecture

<img width="2054" height="8192" alt="5G NR OAI Full-Grid SNN-2026-04-15-063342" src="https://github.com/user-attachments/assets/fd7a7891-5aa2-4c30-9feb-3e19c5dfbb9c" />

---

# 2. Experimental Fairness Principle

To ensure a valid system-level comparison:

- Identical FFT-domain input grids  
- Identical channel realizations  
- Identical noise conditions  
- Identical LDPC decoding backend  

Only the **receiver frontend is replaced**.

### Formal Statement

Both receivers operate on identical FFT-domain samples extracted from the same OAI execution trace under identical channel state and noise realization.

This ensures that all performance differences (BER/LLR behavior) are solely attributable to the receiver design.

---

# 3. Full Pipeline

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
│ - circular carrier reorder │
│ - tensor reshape           │
│ - output [1,1,4,14,1272]   │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ SpikingRxModel             │
│ SEW-ResNet + LIF neurons   │
│ ch → eq → llr prediction   │
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

# 4. Dataset Format (Bundle Structure)

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

# 5. Carrier Reordering (Critical)

OAI FFT indexing follows circular resource mapping:

```python
idx = (np.arange(used_sc) + first_carrier_offset) % n_sc_full
```

### Why necessary

Without this:

- DMRS alignment breaks  
- channel estimation target incorrect  
- LLR supervision invalid  
- BER comparison becomes meaningless  

---

# 6. Input Representation

```
[B, T, C, H, W] = [B, 1, 4, 14, 1272]
```

| Channel | Description |
|--------|------------|
| 0 | Real part |
| 1 | Imag part |
| 2 | DMRS mask |
| 3 | Data mask |

---

# 7. Model Architecture

```
shared spiking encoder
→ channel estimation head
→ equalization head
→ LLR prediction head
```

Outputs:

```
ch  : [B, 2, H, W]
eq  : [B, 2, H, W]
llr : [B, G]
```

---

# 8. LLR Calibration

```
LLR' = LLR / T   (T = 2.0)
```

Ensures compatibility with OAI LDPC decoder.

---

# 9. Training

```bash
python src/train/train_spikingrx_oai.py \
    --dataset ./dataset_oai_bundle \
    --epochs 100 \
    --batch-size 8
```

---

# 10. Inference

```bash
python src/inference/infer_oai_serial.py \
    --bundle example_bundle/sample_0001 \
    --ckpt checkpoints/spikingrx_oai_serial_best.pt \
    --out inferred_llr.bin
```

---

# 11. BER Evaluation

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

# 12. Metric Definition

```
BER = (# erroneous bits) / (total transmitted bits)
```

- computed after LDPC decoding  
- averaged across bundles  

---

# 13. Dataset Selection Policy

- warm-up bundles removed  
- only steady-state data used  
- deterministic selection  

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

Not claimed:

- universal superiority  
- coding gain improvement  

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
- Only processed snapshots and evaluation results are provided  
- All results are reproducible from bundle format  

---

# ================================
# END
# ================================
