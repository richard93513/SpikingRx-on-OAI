# SpikingRx-on-OAI

SpikingRx-on-OAI integrates a spiking neural network-based receiver into the OpenAirInterface (OAI) 5G NR downlink physical layer.

The system performs end-to-end learning on full FFT-domain PDSCH resource grids and outputs log-likelihood ratios (LLRs), which are evaluated through the standard OAI rate recovery and LDPC decoding chain.

---

## Receiver Comparison Setup

This work performs a **controlled receiver-level comparison**.

Instead of comparing two independent systems, we use the **same OAI full-grid FFT data** and evaluate two receiver implementations under identical conditions:

- Original OAI receiver (channel estimation + equalization + demapping)
- SpikingRx receiver (replacing the above modules)

Both outputs are fed into the **same decoding chain**:

- Rate dematching (`rmunmatch_spx`)
- LDPC decoding (`ldpctest_spx`)

The final metric is computed by comparing decoded bits with the original transmitted bits (`txbits`).

### Key Properties

- Identical input data (same full-grid dump)
- Identical channel realization
- Identical decoding process
- Only the receiver is changed

This ensures that performance differences (BER) are solely due to the receiver design.

---

<img width="2054" height="8192" alt="5G NR OAI Full-Grid SNN-2026-04-15-063342" src="https://github.com/user-attachments/assets/fd7a7891-5aa2-4c30-9feb-3e19c5dfbb9c" />


## Full Pipeline

```text
┌────────────────────────────┐
│ OAI gNB / UE (rfsim mode)  │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ fullgrid.bin              │
│ dmrs_mask.bin             │
│ data_mask.bin             │
│ demapper_llr_f32.bin      │
│ txbits.bin                │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ oai_to_spikingrx_tensor.py │
│ circular carrier reorder   │
│ build [1,1,4,14,1272]      │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ SpikingRxModel            │
│ ch → eq → llr             │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ inferred_llr.bin          │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ rmunmatch_spx             │
│ ldpctest_spx              │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ decoded_bits.bin          │
│ BER vs txbits.bin         │
└────────────────────────────┘
```

---

## Repository Layout

```text
SpikingRx-on-OAI/
├── src/
├── oai_change/
├── spx_records/
├── checkpoints/
├── example_bundle/
├── legacy/
```

---

## Purpose of `oai_change/`

This module modifies OpenAirInterface to support:

### 1. PHY-level data extraction

- FFT-domain received grid  
- channel estimates  
- equalized symbols  
- demapper LLR  
- transport block bits  

### 2. Decoder validation tools

- `rmunmatch_spx`  
- `ldpctest_spx`  

### 3. Reproducibility layer

- patch-based OAI modification tracking  

---

## Dataset Format

Each bundle contains:

```text
fullgrid.bin
dmrs_mask.bin
data_mask.bin
demapper_llr_f32.bin
txbits.bin
ldpc_cfg.txt
pdsch_cfg.txt
```

---

## Carrier Reordering (Critical)

OAI FFT indexing is circular and must be explicitly reconstructed:

```python
idx = (np.arange(used_sc) + first_carrier_offset) % n_sc_full
```

Failure to apply correct indexing results in:

- DMRS misalignment  
- incorrect equalization targets  
- invalid BER computation  

---

## Network Input

```text
[B, T, C, H, W] = [B, 1, 4, 14, 1272]
```

| Channel | Description |
|----------|-------------|
| 0 | Real part |
| 1 | Imag part |
| 2 | DMRS mask |
| 3 | Data mask |

---

## Model Architecture

```text
shared spiking encoder
→ channel estimation head
→ equalization head
→ LLR prediction head
```

Outputs:

```text
ch  : [B, 2, 14, 1272]
eq  : [B, 2, 14, 1272]
llr : [B, G]
```

---

## Training

```bash
python src/train/train_oai_serial.py \
    --dataset ./dataset_oai_bundle \
    --epochs 100 \
    --batch-size 8
```

---

## Inference

```bash
python src/inference/infer_oai_serial.py \
    --bundle example_bundle/sample_0001 \
    --ckpt checkpoints/spikingrx_oai_serial_best.pt \
    --out inferred_llr.bin
```

---

## BER Evaluation

```bash
python src/inference/check_oai_llr_decode.py \
    inferred_llr.bin \
    example_bundle/sample_0001/ldpc_cfg.txt \
    example_bundle/sample_0001/pdsch_cfg.txt \
    example_bundle/sample_0001/txbits.bin \
    --rmunmatch ./oai_change/rmunmatch_spx \
    --ldpctest ./oai_change/ldpctest_spx
```

### Evaluation chain

```text
LLR
→ rate de-matching
→ LDPC decoding
→ transport block comparison
→ BER
```

---

## Experimental Results

All results are stored under:

```text
spx_records/snapshots_snr/
```

### Reported outputs

#### 1. BER vs SNR curves

- SpikingRx BER vs SNR
- OAI demapper BER vs SNR
- Direct comparison under identical channel realizations

#### 2. LLR-level analysis

- Correlation vs SNR
- Distribution consistency
- Case-wise scatter evaluation

#### 3. Case-wise evaluation

- high-SNR regime
- waterfall (cliff) region
- low-SNR regime

---

## BER Reporting Convention

BER is computed at transport block level:

```text
BER = (# of erroneous bits) / (total transmitted bits)
```

Typical behavior:

- High SNR: BER ≈ 0  
- Transition region: rapid increase (waterfall effect)  
- Low SNR: BER saturates (~0.2–0.5 depending on coding rate)

---

## Output Artifacts

Generated during evaluation:

- BER vs SNR summary tables  
- SpikingRx vs OAI comparison CSV  
- LLR statistics per SNR  
- Case study reports  

---

## Limitations

- QPSK modulation only  
- Single code block (`C = 1`)  
- Single slot processing (`T = 1`)  
- AWGN channel only  

---

## Future Work

- Extension to higher-order modulation (16QAM / 64QAM)  
- Multi-codeblock decoding  
- BLER evaluation  
- Temporal modeling across slots  
- Comparison with classical MMSE receivers  
- Hardware / neuromorphic deployment

---

## Dataset (Google Drive)

Due to the size and file granularity of the raw OAI dumps, the dataset is distributed as a compressed archive via Google Drive:

```
https://drive.google.com/file/d/1vO04jncqe-hFHiepl01yRuGgezznBQ16/view?usp=sharing
```

---

### Contents

The archive contains processed experimental outputs located under:

```text
spx_records/snapshots_snr/
```

The provided data includes:

- BER vs SNR evaluation results (SpikingRx and OAI baseline)
- Noise sweep summaries
- LLR statistical analysis (correlation, histogram, scatter)
- Case-wise evaluation across different SNR regimes

---

### Notes

- Raw bundle data (e.g., `*.bin`, `bundle_noise_power_*`) is not included due to storage constraints.
- Only processed results and visualization outputs are provided.
- The dataset is packaged as a `.tar.gz` archive for efficient distribution.

---

### Extraction

```bash
tar -xzvf snapshots_snr.tar.gz
```
