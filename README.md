# SpikingRx-on-OAI

SpikingRx-on-OAI integrates a spiking neural receiver with the OpenAirInterface (OAI) 5G NR downlink PHY.

The repository trains a spiking model directly on full FFT-domain OAI PDSCH resource grids, predicts channel / equalized symbols / LLRs, and evaluates the final result using the exact OAI rate-unmatching and LDPC decoder chain.

---

## What This Repository Does

Traditional learning-based receivers often stop at one of the following:

* Channel estimation MSE
* Equalized symbol error
* LLR correlation against a reference demapper

This repository instead evaluates the complete receiver pipeline:

```text
OAI full-grid FFT dump
→ occupied-carrier reorder
→ spiking neural receiver
→ predicted LLR
→ OAI rate unmatching
→ OAI LDPC decoding
→ transport-block BER
```

Therefore, the final metric is not intermediate reconstruction quality, but decoded transport-block BER.

---

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
├── README.md
├── requirements.txt
├── docs/
├── src/
├── checkpoints/
├── example_bundle/
├── legacy/
└── oai_change/
    ├── CMakeLists.txt
    ├── patches/
    │   └── oai_local_changes.patch
    ├── openair1/
    │   └── PHY/
    │       ├── CODING/
    │       │   └── TESTBENCH/
    │       │       ├── rmunmatch_spx.c
    │       │       ├── ldpctest_spx.c
    │       │       └── spx_ldpc_test.c
    │       ├── NR_UE_ESTIMATION/
    │       │   └── nr_dl_channel_estimation.c
    │       ├── NR_UE_TRANSPORT/
    │       │   ├── nr_dlsch_demodulation.c
    │       │   └── nr_dlsch_decoding.c
    │       └── NR_TRANSPORT/
    │           └── nr_dlsch_coding.c
    ├── radio/rfsimulator/
    │   └── apply_channelmod.c
    └── targets/PROJECTS/GENERIC-NR-5GC/CONF/
```

### Main Folders

| Folder            | Purpose                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `src/`            | Main training, inference, dataset, and model code                                                    |
| `oai_change/`     | Local OpenAirInterface modifications used to dump internal PHY data and run exact decoder validation |
| `docs/`           | Figures, BER curves, notes, screenshots                                                              |
| `checkpoints/`    | Trained model weights                                                                                |
| `example_bundle/` | Minimal example of required OAI dump files                                                           |
| `legacy/`         | Old compressed-grid experiments and deprecated versions                                              |

### Purpose of `oai_change/`

The `oai_change/` folder contains the OpenAirInterface source modifications required to generate the dataset and evaluate the neural receiver.

These changes are used for three purposes:

1. Dump internal PHY tensors from OAI UE processing:

   * full FFT-domain received grid
   * channel estimate
   * equalized symbols
   * demapper LLRs
   * transport-block bits

2. Add exact decoder-side validation tools:

   * `rmunmatch_spx`
   * `ldpctest_spx`

3. Provide a reproducible modified OAI tree and patch file.

Important modified files include:

| File                         | Purpose                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `nr_dl_channel_estimation.c` | Dump FFT-domain received grid and channel estimation results |
| `nr_dlsch_demodulation.c`    | Dump equalized symbols and demapper LLRs                     |
| `nr_dlsch_decoding.c`        | Export decoder-side intermediate data                        |
| `nr_dlsch_coding.c`          | Export transmitted transport-block bits                      |
| `rmunmatch_spx.c`            | Convert predicted LLRs into OAI LDPC decoder input format    |
| `ldpctest_spx.c`             | Run standalone LDPC decoding on the predicted LLRs           |
| `oai_local_changes.patch`    | Patch file for reproducing all OAI modifications             |

To rebuild the decoder-side validation tools:

```bash
cd oai_change/openair1/PHY/CODING/TESTBENCH
make
```

This produces:

```text
rmunmatch_spx
ldpctest_spx
```

## Example Dataset Bundle

Each sample bundle should contain:

```text
example_bundle/sample_0001/
├── fullgrid.bin
├── dmrs_mask.bin
├── data_mask.bin
├── demapper_llr_f32.bin
├── txbits.bin
├── ldpc_cfg.txt
├── pdsch_cfg.txt
├── ue_tb.bin
├── ue_c_seg00.bin
└── ...
```

Do not upload large raw datasets into the repository. Only include a small example folder or a text tree showing the required files.

---

## Critical Preprocessing: Carrier Reordering

OAI stores the complete FFT buffer in `fullgrid.bin`.
The occupied NR carriers wrap around the FFT index and cannot be extracted with a simple slice.

Incorrect:

```python
grid_used = grid_full[:, first_sc:first_sc+used_sc]
```

Correct:

```python
first_carrier_offset = n_sc_full - (used_sc // 2)
idx = (np.arange(used_sc) + first_carrier_offset) % n_sc_full
grid_used = grid_full[:, idx]
```

Without this circular reorder:

* DMRS mask becomes misaligned
* Heatmaps appear shifted
* Equalization targets become incorrect
* BER evaluation fails

---

## Network Input

The current model input tensor is:

```text
[B, T, C, H, W] = [B, 1, 4, 14, 1272]
```

Input channels:

| Channel | Meaning                         |
| ------- | ------------------------------- |
| 0       | Real part of received grid      |
| 1       | Imaginary part of received grid |
| 2       | DMRS mask                       |
| 3       | Data mask                       |

---

## Model Architecture

The model uses a serial prediction structure:

```text
shared spiking encoder
→ channel head
→ equalization head
→ LLR head
```

Detailed dependency:

```text
feat
→ ch_head(feat)
→ eq_head(feat + ch_pred)
→ llr_head(feat + ch_pred + eq_pred)
```

Outputs:

```text
pred["ch"]  : [B, 2, 14, 1272]
pred["eq"]  : [B, 2, 14, 1272]
pred["llr"] : [B, G]
```

Current assumptions:

* QPSK only
* `bits_per_symbol = 2`
* `T = 1`
* `C = 1`

LLR bit order:

```text
RE0(bit0, bit1), RE1(bit0, bit1), ...
```

---

## Installation

```bash
git clone https://github.com/richard93513/SpikingRx-on-OAI.git
cd SpikingRx-on-OAI
pip install -r requirements.txt
```

If using GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## How to Train

```bash
python src/train/train_oai_serial.py \
    --dataset ./dataset_oai_bundle \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4
```

Typical output:

```text
checkpoints/spikingrx_oai_serial_best.pt
```

Recommended loss structure:

```text
loss = w_ch * L_ch + w_eq * L_eq + w_llr * L_llr
```

with `w_llr` as the dominant term.

---

## How to Run Inference

```bash
python src/inference/infer_oai_serial.py \
    --bundle example_bundle/sample_0001 \
    --ckpt checkpoints/spikingrx_oai_serial_best.pt \
    --out inferred_llr.bin
```

This command:

1. Loads the OAI bundle
2. Builds the `[1,1,4,14,1272]` tensor
3. Runs `SpikingRxModel`
4. Writes predicted LLRs to `inferred_llr.bin`

---

## How to Evaluate BER

```bash
python src/inference/check_oai_llr_decode.py \
    inferred_llr.bin \
    example_bundle/sample_0001/ldpc_cfg.txt \
    example_bundle/sample_0001/pdsch_cfg.txt \
    example_bundle/sample_0001/txbits.bin \
    --rmunmatch ./oai_change/rmunmatch_spx \
    --ldpctest ./oai_change/ldpctest_spx \
    --llr-scale 1.0
```

The script performs:

```text
inferred_llr.bin
→ rmunmatch_spx
→ rm_exact_spx_seg00_i8.bin
→ ldpctest_spx
→ decoded_bits.bin
→ compare against txbits.bin
→ BER
```

Expected output:

```text
[RESULT] BER=0.012500 bit_errors=13/1040
```

---

## LLR Magnitude Tuning

The LDPC decoder is highly sensitive to LLR scale.

Try:

```bash
--llr-scale 0.5
--llr-scale 1.0
--llr-scale 2.0
--llr-scale 4.0
```

because the effective decoder input is approximately:

```text
network_output / llr_temperature × llr_scale
```

---

## Output Metadata

`check_oai_llr_decode.py` writes `oai_decode_meta.json`:

```json
{
  "A": 1056,
  "G": 14400,
  "C": 1,
  "llr_scale": 1.0,
  "ber": 0.023,
  "bit_errors": 24,
  "rmunmatch_rc": 0,
  "ldpctest_rc": 0
}
```

---

## Current Limitations

* QPSK only
* Single code block (`C = 1`)
* Only `seg00` is decoded
* Single slot (`T = 1`)
* No 16QAM / 64QAM / 256QAM support yet

---

## Future Work

* Multi-codeblock support (`C > 1`)
* Higher-order modulation support
* BLER vs SNR evaluation
* Multi-slot temporal input (`T > 1`)
* Comparison against classical MMSE receivers
* More complete OAI patch automation
* FPGA or neuromorphic deployment
