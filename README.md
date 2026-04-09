# SpikingRx for OAI NR PDSCH Decoding

SpikingRx is a spiking-neural-network receiver pipeline trained on full OpenAirInterface (OAI) 5G NR downlink resource grids.

The model takes the complete FFT-domain OFDM grid, predicts channel estimates, equalized symbols, and log-likelihood ratios (LLRs), then evaluates performance through the exact OAI rate-matching and LDPC decoding chain.

---

## Overview

This repository implements the following end-to-end pipeline:

```text
OAI full FFT grid
→ occupied-carrier reorder
→ spiking encoder
→ channel prediction
→ equalization prediction
→ LLR prediction
→ OAI rate unmatching
→ OAI LDPC decoding
→ BER evaluation
```

Unlike earlier experiments that used compressed 32×64 grids or only compared MSE against OAI intermediate outputs, this version uses:

* Full rectangular OFDM grids: `[14, 1272]`
* Full FFT-domain carrier extraction from OAI `fullgrid.bin`
* Exact OpenAirInterface decoding tools:

  * `rmunmatch_spx`
  * `ldpctest_spx`
* Final metric: transport-block BER against the original transmitted bits

---

## Repository Structure

```text
src/
├── datasets/
│   └── dataset_oai_bundle.py
├── models/
│   ├── spikingrx_model.py
│   ├── sew_block.py
│   ├── conv_block.py
│   ├── norm_layer.py
│   └── lif_neuron.py
├── train/
│   └── train_oai_serial.py
├── inference/
│   ├── infer_oai_serial.py
│   └── check_oai_llr_decode.py
└── utils/
    ├── oai_to_spikingrx_tensor.py
    └── visualize_fullgrid.py
```

Legacy scripts from the older compressed-grid pipeline should be moved into a separate `legacy/` folder.

---

## Dataset Format

Each OAI sample bundle should contain:

```text
dataset_oai_bundle/
├── fullgrid.bin
├── dmrs_mask.bin
├── data_mask.bin
├── demapper_llr_f32.bin
├── txbits.bin
├── ldpc_cfg.txt
├── pdsch_cfg.txt
└── ...
```

### Input Tensor

The final network input is:

```text
[B, T, C, H, W] = [B, 1, 4, 14, 1272]
```

where the 4 channels are:

1. Real part of received grid
2. Imaginary part of received grid
3. DMRS mask
4. Data RE mask

---

## Carrier Reordering

The most important preprocessing step is converting the OAI full FFT dump into the actual occupied NR carrier region.

Incorrect older code:

```python
grid_full[:, first_sc:first_sc+used_sc]
```

Correct implementation:

```python
first_carrier_offset = n_sc_full - (used_sc // 2)
idx = (np.arange(used_sc) + first_carrier_offset) % n_sc_full
grid_used = grid_full[:, idx]
```

This circular reorder is required because the occupied NR carriers wrap around the FFT buffer.

Without this step:

* Heatmaps appear shifted
* DMRS and data masks do not align
* LLR prediction fails
* BER becomes meaningless

---

## Model Architecture

`SpikingRxModel` uses a serial prediction pipeline:

```text
shared spiking encoder
→ channel head
→ equalization head
→ LLR head
```

More specifically:

```text
feat
→ ch_head(feat)
→ eq_head(feat + ch_pred)
→ llr_head(feat + ch_pred + eq_pred)
```

### Outputs

```text
pred["ch"]  : [B, 2, 14, 1272]
pred["eq"]  : [B, 2, 14, 1272]
pred["llr"] : [B, G]
```

Current implementation assumptions:

* QPSK only
* `bits_per_symbol = 2`
* `T = 1`

The LLR ordering is:

```text
RE0(bit0, bit1), RE1(bit0, bit1), ...
```

This ordering must exactly match the expected input ordering of `rmunmatch_spx`.

---

## Training

Example:

```bash
python src/train/train_oai_serial.py \
    --dataset dataset_oai_bundle \
    --epochs 100 \
    --batch-size 8
```

Training target:

* Channel estimate
* Equalized symbols
* OAI demapper LLRs

Recommended primary objective:

```text
loss = w_ch * L_ch + w_eq * L_eq + w_llr * L_llr
```

where `L_llr` should dominate because the final decoding performance is determined mainly by LLR quality.

---

## Inference

Example:

```bash
python src/inference/infer_oai_serial.py \
    --bundle dataset_oai_bundle/sample_0001 \
    --ckpt checkpoints/spikingrx_oai_serial_best.pt
```

The inference script should:

1. Load the OAI bundle
2. Build the `[1,1,4,14,1272]` tensor
3. Run `SpikingRxModel`
4. Save predicted LLRs to a binary file

---

## BER Evaluation Through OAI Decoder

To evaluate predicted LLRs:

```bash
python src/inference/check_oai_llr_decode.py \
    inferred_llr.bin \
    ldpc_cfg.txt \
    pdsch_cfg.txt \
    txbits.bin \
    --rmunmatch ./rmunmatch_spx \
    --ldpctest ./ldpctest_spx
```

This script performs:

```text
predicted LLR
→ rmunmatch_spx
→ ldpctest_spx
→ decoded_bits.bin
→ compare with txbits.bin
→ BER
```

---

## LLR Scale Tuning

The decoder is sensitive to LLR magnitude.

Useful values to sweep:

```text
--llr-scale 0.5
--llr-scale 1.0
--llr-scale 2.0
--llr-scale 4.0
```

The effective decoder input magnitude is approximately:

```text
network_output / llr_temperature × llr_scale
```

---

## Output Metadata

`check_oai_llr_decode.py` writes a JSON file:

```json
{
  "A": 1056,
  "G": 14400,
  "C": 1,
  "llr_scale": 1.0,
  "ber": 0.023,
  "bit_errors": 24
}
```

This metadata also includes:

* Decoder return codes
* File statistics
* Oracle comparison results
* Paths used during the run

---

## Current Limitations

* QPSK only
* Currently assumes `C = 1`
* Only `seg00` is decoded in the LDPC stage
* Single-slot input (`T = 1`)
* No support yet for 16QAM / 64QAM / 256QAM

---

## Future Work

* Multi-codeblock support (`C > 1`)
* Higher-order modulation
* BLER vs SNR evaluation
* Temporal input (`T > 1`)
* Compare against classical MMSE receiver
* Export trained model for FPGA / neuromorphic deployment

---
