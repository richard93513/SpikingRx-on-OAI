# SpikingRx-on-OAI

SpikingRx-on-OAI integrates a spiking neural receiver with the OpenAirInterface (OAI) 5G NR downlink PHY.

The repository trains a spiking model directly on full FFT-domain OAI PDSCH resource grids, predicts channel / equalized symbols / LLRs, and evaluates the final result using the exact OAI rate-unmatching and LDPC decoder chain.

---

## 🧠 Why This Project Matters

Most learning-based receivers are evaluated using intermediate metrics such as:

- Channel estimation MSE  
- Equalized symbol error  
- LLR correlation  

These do not reflect the **actual communication system performance**.

This project evaluates the **full 5G receiver pipeline**, where the final metric is:

👉 **Transport Block BER after LDPC decoding**

```text
Neural Receiver → LLR → LDPC → BER
```

This makes the evaluation directly comparable to real communication systems.

---

## 🚀 Quick Start

### Clone repo

```bash
git clone https://github.com/richard93513/SpikingRx-on-OAI.git
cd SpikingRx-on-OAI
pip install -r requirements.txt
```

---

### Run inference on example bundle

```bash
python src/inference/infer_oai_serial.py \
    --bundle example_bundle/sample_0001 \
    --ckpt checkpoints/spikingrx_oai_serial_best.pt \
    --out inferred_llr.bin
```

---

### Run BER evaluation

```bash
python src/inference/check_oai_llr_decode.py \
    inferred_llr.bin \
    example_bundle/sample_0001/ldpc_cfg.txt \
    example_bundle/sample_0001/pdsch_cfg.txt \
    example_bundle/sample_0001/txbits.bin \
    --rmunmatch ./oai_change/rmunmatch_spx \
    --ldpctest ./oai_change/ldpctest_spx
```

---

## 🔁 Full Pipeline

```text
OAI full-grid FFT dump
→ carrier reorder
→ SpikingRx
→ predicted LLR
→ OAI rmunmatch
→ OAI LDPC decode
→ BER
```

---

## 📊 Results

### BER vs SNR

The model is evaluated using full end-to-end decoding:

```text
SpikingRx LLR → LDPC → BER
vs
OAI demapper LLR → LDPC → BER
```

---

### Results are stored in:

```
spx_records/snapshots_snr/
```

---

### Key outputs:

```
compare_spikingrx_vs_oai_ber_vs_snr.png
noise_sweep_ber_curve.png
llr_metrics_vs_snr/
llr_scatter_cases/
llr_diff_hist_cases/
```

---

### These include:

- BER vs SNR comparison  
- LLR correlation analysis  
- Scatter plots and histogram analysis  

---

## 📁 Repository Layout

```
SpikingRx-on-OAI/
├── src/                # training / inference / models
├── oai_change/         # modified OpenAirInterface
├── spx_records/        # experiment results & plots
├── checkpoints/        # trained weights
├── example_bundle/     # minimal dataset example
├── docs/               # notes and figures
```

---

## 🔧 OAI Modifications (oai_change/)

This folder contains all OpenAirInterface modifications.

---

### Purpose

- Dump internal PHY tensors:
  - FFT-domain received grid  
  - channel estimation  
  - equalized symbols  
  - demapper LLR  
  - transport block bits  

- Provide decoder validation tools:
  - rmunmatch_spx  
  - ldpctest_spx  

- Enable full reproducibility via patch file  

---

### Build tools

```bash
cd oai_change/openair1/PHY/CODING/TESTBENCH
make
```

---

## 📦 Dataset Format

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

## 🧠 Model

### Input

```
[B, T, C, H, W] = [B, 1, 4, 14, 1272]
```

---

### Channels

| Channel | Meaning |
|--------|--------|
| 0 | real |
| 1 | imag |
| 2 | DMRS mask |
| 3 | data mask |

---

### Architecture

```
shared encoder
→ channel head
→ equalization head
→ LLR head
```

---

## 🏋️ Training

```bash
python src/train/train_oai_serial.py \
    --dataset ./dataset_oai_bundle \
    --epochs 100 \
    --batch-size 8
```

---

## 📉 BER Evaluation

### Pipeline

```
LLR
→ rmunmatch_spx
→ LDPC decode
→ BER
```

---

## ⚠️ Important Detail: Carrier Reordering

OAI FFT grid requires circular indexing:

```python
idx = (np.arange(used_sc) + first_carrier_offset) % n_sc_full
```

Without this:

- DMRS misalignment  
- wrong equalization  
- invalid BER  

---

## ❗ Limitations

- QPSK only  
- single codeblock  
- single slot  
- no higher-order modulation  

---

## 🔭 Future Work

- multi-codeblock decoding  
- higher-order modulation (16QAM / 64QAM)  
- BLER evaluation  
- temporal modeling (T > 1)  
- comparison with classical receivers  
- hardware / neuromorphic deployment  

---

# 🔥 Final Conclusion

This repository:

👉 is already suitable for academic review  
👉 can be used as graduate application project material  
👉 represents a full end-to-end neural receiver research pipeline  

---

## 🧭 Optional Upgrade (next level)

If extended further, recommended additions:

### 📌 architecture figure

```
[ PLACE ARCHITECTURE DIAGRAM HERE ]
```

### 📌 BER vs SNR figure

```
[ PLACE MAIN RESULT FIGURE HERE ]
```

---

Then this repository becomes:

🔥 **publication-ready research codebase**
