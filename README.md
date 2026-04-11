# SpikingRx-on-OAI

SpikingRx-on-OAI integrates a spiking neural network-based receiver into the OpenAirInterface (OAI) 5G NR downlink physical layer.

The system performs end-to-end learning on full FFT-domain PDSCH resource grids and outputs log-likelihood ratios (LLRs), which are evaluated through the standard OAI rate recovery and LDPC decoding chain.

---

## 1. Motivation

Conventional learning-based wireless receivers are typically evaluated using proxy metrics such as:

- Channel estimation mean squared error (MSE)
- Symbol-level error after equalization
- LLR correlation or distribution similarity

However, these metrics do not directly reflect system-level communication performance.

This project evaluates receiver performance using the final communication metric:

> **Transport Block Error Rate (BER) after LDPC decoding**

The evaluation is fully aligned with standard 5G NR receiver behavior:

```
Receiver → LLR → LDPC decoding → BER
```

This enables a direct comparison between neural receivers and the conventional OAI demapper-based receiver.

---

## 2. System Overview

The complete processing chain is:

```
OAI full-grid FFT output
→ carrier reordering
→ SpikingRx inference
→ LLR estimation
→ rate de-matching (OAI)
→ LDPC decoding (OAI)
→ BER computation
```

A baseline system is defined as:

```
OAI demapper LLR → LDPC decoding → BER
```

Both systems are evaluated under identical channel realizations and dataset conditions.

---

## 3. Repository Structure

```
SpikingRx-on-OAI/
├── src/                # training, inference, evaluation
├── oai_change/         # modified OpenAirInterface modules
├── spx_records/        # experimental outputs and logs
├── checkpoints/        # trained model parameters
├── example_bundle/     # minimal runnable dataset
├── docs/               # documentation and notes
```

---

## 4. Dataset Format

Each data bundle corresponds to one transmission instance and contains:

```
fullgrid.bin
dmrs_mask.bin
data_mask.bin
demapper_llr_f32.bin
txbits.bin
ldpc_cfg.txt
pdsch_cfg.txt
```

These files represent:

- FFT-domain received resource grid
- DMRS and data allocation masks
- Baseline demapper LLR outputs
- Ground-truth transmitted bits
- Channel coding configuration

---

## 5. Model Specification

### Input Tensor

```
[B, T, C, H, W] = [B, 1, 4, 14, 1272]
```

### Channel Definition

| Channel | Description |
|----------|-------------|
| 0 | Real part of received signal |
| 1 | Imaginary part |
| 2 | DMRS allocation mask |
| 3 | Data allocation mask |

### Architecture

The model follows a modular design:

```
Shared encoder
→ Channel estimation head
→ Equalization head
→ LLR prediction head
```

The output is a soft bit representation compatible with the OAI LDPC decoding pipeline.

---

## 6. Training

```bash
python src/train/train_oai_serial.py \
    --dataset ./dataset_oai_bundle \
    --epochs 100 \
    --batch-size 8
```

Training is performed on full FFT-domain resource grids with supervised learning on reference LLRs.

---

## 7. Inference

```bash
python src/inference/infer_oai_serial.py \
    --bundle example_bundle/sample_0001 \
    --ckpt checkpoints/spikingrx_oai_serial_best.pt \
    --out inferred_llr.bin
```

---

## 8. BER Evaluation

```bash
python src/inference/check_oai_llr_decode.py \
    inferred_llr.bin \
    example_bundle/sample_0001/ldpc_cfg.txt \
    example_bundle/sample_0001/pdsch_cfg.txt \
    example_bundle/sample_0001/txbits.bin \
    --rmunmatch ./oai_change/rmunmatch_spx \
    --ldpctest ./oai_change/ldpctest_spx
```

The evaluation follows the standard OAI decoding chain:

```
LLR → rate de-matching → LDPC decoding → BER
```

---

## 9. Carrier Reordering

OAI FFT grids use circular frequency indexing. Proper alignment is required:

```python
idx = (np.arange(used_sc) + first_carrier_offset) % n_sc_full
```

Incorrect reordering leads to:

- DMRS misalignment  
- incorrect channel estimation  
- invalid BER computation  

---

## 10. Experimental Outputs

All evaluation results are stored under:

```
spx_records/snapshots_snr/
```

Typical outputs include:

- BER vs SNR evaluation
- SpikingRx vs OAI performance comparison
- LLR statistical analysis
- Case-wise performance breakdown

---

## 11. Evaluation Scope

The current implementation is limited to:

- QPSK modulation
- Single codeblock transmission
- Single OFDM slot processing
- AWGN channel model

---

## 12. Future Work

Planned extensions include:

- Support for higher-order modulation schemes (16QAM / 64QAM)
- Multi-codeblock decoding
- Block error rate (BLER) evaluation
- Temporal sequence modeling across multiple slots
- Comparison with classical MMSE-based receivers
- Hardware / neuromorphic deployment

---

## 13. Summary

This repository implements a full end-to-end neural receiver integrated into a standard 5G NR PHY stack. The evaluation is performed at the system level (post-LDPC BER), enabling direct comparison between learned and conventional demodulation pipelines under identical channel conditions.
