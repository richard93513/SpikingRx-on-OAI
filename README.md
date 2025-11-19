# SpikingRx-on-OAI

A complete integration of the **SpikingRx** spiking neural receiver with **OpenAirInterface (OAI)** full-grid PDSCH dumps.
This project demonstrates how a spiking neural network (SNN) can replace conventional 5G NR receiver blocks (equalizer + demapper) using the method proposed in:

> Gupta, A. et al., “SpikingRx: From Neural to Spiking Receiver”, 2025.

---

## 1. Project Overview

This repository implements the full physical-layer pipeline:

```
OAI gNB/UE (rfsim mode)
      ↓
Full-grid PDSCH FFT dump (/tmp/spx_fullgrid_*)
      ↓
load_oai_fullgrid() → 32×32×T Tensor
      ↓
SpikingRx Model (StemConv + 6×SEW + LIF + Readout)
      ↓
LLR output + Spike activity visualizations
```

The goal is to evaluate whether a spiking neural receiver can interpret **real 5G NR demodulated data** produced by OAI.

---

## 2. What is SpikingRx?

SpikingRx replaces the classical 5G NR receiver pipeline:

### Traditional Receiver
- Channel Estimation 
- Equalization (LMMSE / ZF) 
- Soft Demapping 

### SpikingRx Receiver
A **single, end-to-end spiking neural network module**:
- SEW-ResNet spiking blocks 
- LIF neurons with triangular surrogate gradient 
- Time-unrolled spiking dynamics 
- ANN readout generating soft bits (LLR) 

This enables:
- Neuromorphic efficiency 
- End-to-end learnability 
- Lower computational energy 
- Compatibility with LDPC decoder input 

---

## 3. Repository Structure

```
SpikingRx-on-OAI/
│
├── checkpoints/                # trained SNN models
│
├── docs/
│   ├── images/                 # system diagrams & architecture images
│   ├── notes/                  # development notebook & logs
│   ├── papers/                 # SpikingRx references (optional)
│   └── results/
│       ├── train/              # training curves
│       ├── inference/          # OAI inference results
│       └── visualize/          # GIF animations & comparisons
│
├── src/
│   ├── data/                   # OAI full-grid loader
│   ├── models/                 # SpikingRx (LIF + SEW + Stem)
│   ├── train/                  # synthetic OFDM training pipeline
│   ├── inference/              # run SNN on OAI dumps
│   ├── visualize/              # spike/LLR visualization tools
│   └── tests/                  # unit tests for modules
│
├── requirements.txt
└── README.md                   # (this document)
```

---

## 4. Training the SpikingRx Model (Synthetic QPSK)

Training uses a synthetic QPSK OFDM grid for supervised learning.

Run:

```
cd src/train
python train_spikingrx.py
```

Outputs:

- checkpoints/spikingrx_checkpoint.pth 
- docs/results/train/loss_curve.png 
- docs/results/train/loss_history.npy 

### Training parameters:

- Grid size: 32×32 
- Timesteps: T = 5 
- Modulation: QPSK 
- Loss: CrossEntropy (bit-wise) 
- Surrogate gradient: Triangular 
- Neuron: LIF (beta = 0.9, theta = 0.5) 
- Optimizer: Adam 
---

## 5. Running Inference on OAI Dumps

OAI must generate full-grid FFT dumps:

```
/tmp/spx_fullgrid_f*_s*.bin
```

Run inference using the trained model:

```
cd src/inference
python run_spikingrx_on_oai_dump.py \
    --ckpt ../../checkpoints/spikingrx_checkpoint.pth
```

Outputs saved to:

```
docs/results/inference/
    llr_heatmap.png
    llr.npy
    spike_rate.png
```

This produces:

- Time-averaged spike rate 
- LLR heatmap for bit0 & bit1 
- Saved numpy LLR tensor 

---

## 6. Visualization (Spike Activity & Training Comparison)

To compare **random-initialized SNN** vs **trained SNN**, run:

```
cd src/visualize
python visualize_spiking_activity.py \
    --ckpt ../train/spikingrx_checkpoint.pth
```

Outputs saved under:

```
docs/results/visualize/
    spike_stage1.gif
    spike_stage2.gif
    spike_stage3.gif
    spike_stage4.gif
    spike_stage5.gif
    spike_stage6.gif
    spike_rate_before_after.png
    llr_heatmap_trained.png
```

### Generated items:

- **GIF animations** of spike maps across T timesteps 
- **Spike-rate comparison** (before vs. after training) 
- **LLR heatmap** showing trained SNN output 
- Perfect for analysis, presentation, and debugging 

---

## 7. Debug Pipeline (Optional)

To inspect internal feature maps (mean/std/shape of each stage):

```
cd src/inference
python run_spikingrx_on_oai_dump_debug.py \
    --ckpt ../../checkpoints/spikingrx_checkpoint.pth
```

This mode prints:

- StemConv output stats 
- Each SEW block output stats 
- Final spike rates 
- LLR statistics 

## 8. How OAI Dumps Are Processed (load_oai_fullgrid)

`load_oai_fullgrid()` converts the raw OAI FFT dump into a spiking-network compatible tensor.

Processing steps:

1. **Read OAI full-grid FFT dump** 
   - File format: complex16 (int16 I/Q) 
   - Shape: [13 OFDM symbols, 1272 subcarriers] for PDSCH region

2. **Extract PDSCH RBs** 
   - Example: 106 PRBs × 12 SC = 1272 used SC 
   - Removes guard bands & unused RE

3. **Convert into float32 real-valued tensors** 
   - Normalize I/Q amplitude 
   - Handle two’s complement sign correctly

4. **Reshape to 32×32 block** 
   - Center crop or interpolate 
   - Matches SpikingRx input resolution

5. **Replicate across T timesteps** 
   - Shape becomes:

```
(1, T=5, 2, 32, 32)
```

6. **Send to GPU/CPU device**

This tensor is then forwarded into SpikingRx for inference.

---

## 9. SpikingRx Model Architecture

### High-level structure:

```
Input (1×T×2×32×32)
      ↓
StemConv
      ↓
SEW Block × 6
      ↓
Temporal Spike Aggregation
      ↓
ANN Readout (1×1 Convs)
      ↓
LLR output (32×32×2)
```

### Components:

- **StemConv** 
  - Conv2D 
  - SpikeNorm 
  - LIF neuron

- **SEW Block** 
  - Spike-Element-Wise residual block 
  - Temporal LIF activation 
  - Supports binary spikes

- **LIF Neuron** 
  - Membrane update: 
    `U[t] = β U[t-1] + (1-β) I[t]` 
  - Spike event: 
    `S[t] = H(U[t] - θ)` 
  - Surrogate gradient: triangular 
  - Soft reset: 
    `U[t] = U[t] - S[t] θ`

- **Readout ANN** 
  - 1×1 Conv (feature reduction) 
  - 1×1 Conv (bit-wise logits) 
  - LLR temperature scaling

Final output shape:

```
(B, 32, 32, 2)   # bit0, bit1
```

---

## 10. Example Outputs (Stored in docs/results)

- **Training**
  - `loss_curve.png`
  - `loss_history.npy`

- **Inference**
  - `llr_heatmap.png`
  - `spike_rate.png`
  - `llr.npy`

- **Visualization**
  - `spike_stage1.gif` ~ `spike_stage6.gif`
  - `spike_rate_before_after.png`
  - `llr_heatmap_trained.png`

These outputs provide a comprehensive view of:
- Model performance 
- Spiking activity 
- Training improvement 
- LLR behavior on real OAI data 

---

## 11. Requirements

```
torch
numpy
matplotlib
tqdm
pillow       # for GIF animation
```

---

## 12. Future Work

- Multi-QAM (16QAM / 64QAM) spiking receiver 
- Training with real OAI demodulated grids 
- BER/BLER comparison against classical NR receiver 
- SIMO/MIMO extension 
- Online learning with variable SNR 
- Replace ANN Readout with fully spiking readout 
- Integration with LDPC decoding 

---

## 13. Author

Richard (richard93513) 
NTUST EE — Communications Systems 
SpikingRx × OAI Integration Project (2025)

---

