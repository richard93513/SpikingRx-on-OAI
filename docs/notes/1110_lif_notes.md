# SpikingRx and LIF Module Notes (System → Modules → Code)

---

## 1. Overall Architecture of SpikingRx

The core goal of **SpikingRx** is to replace the traditional receiver with a *neuralized receiver* (Neural Receiver).

Instead of separating:

- Channel Estimation  
- Equalization  
- Soft Demapping  

SpikingRx integrates them into a single **end-to-end trainable Spiking Neural Network (SNN)**.

The receiver takes **frequency-domain symbols** (after CP removal + FFT) and outputs **LLRs** (Log-Likelihood Ratios) before LDPC decoding.

**End-to-end pipeline:**

**Transmitter:**  
LDPC → Modulation → IFFT → Add CP → Channel  

**Receiver:**  
Remove CP → FFT → SpikingRx → LLR → Decoder  

---

## 2. Network Structure of SpikingRx

SpikingRx is constructed from multiple **SEW-ResNet blocks**.

**Each block contains:**

```
Input (FFT grid)
 ├─ Conv2d (3×3, padding=1)
 ├─ Normalization
 ├─ LIF Spiking Neuron
 └─ Shortcut (Residual Path)
 ↓
ANN Readout → LLR output
```

- The **Conv + Norm** extract local channel features in the frequency grid.  
- The **LIF layer** models temporal spiking behavior over timesteps.  
- The **ANN readout** aggregates multiple timesteps to produce final LLRs.

---

## 3. Implementation-Level Program Modules

SpikingRx is divided into modular Python files:

| Module | Function | Related Paper Section |
|--------|----------|-----------------------|
| `conv_block.py` | Convolution layer + weight initialization | SEW-ResNet Conv part |
| `norm_layer.py` | Normalization layers | Stabilizes membrane potential |
| `lif_neuron.py` | LIF neuron dynamics | Eq. (14), Training details |
| `sew_block.py` | Conv + Norm + LIF + Shortcut | SEW-ResNet Block |
| `spikingrx_model.py` | Full model + ANN readout | Section III main architecture |

`lif_neuron.py` serves as the **fundamental spiking unit**,  
responsible for integrating membrane potentials and producing spikes over time.

---

## 4. Position of the LIF Module Inside the Architecture

Within each SEW-ResNet block:

```
Conv → Norm → [ LIF ] → Shortcut
```

The LIF module receives the processed feature map and applies temporal integration, thresholding, and reset.

---

## 5. Core Computation Steps of the LIF Neuron

The LIF neuron performs **three essential actions**:

---

### 1️⃣ Integrate

\[
U = \beta U + (1 - \beta) I
\]

- `U`: membrane potential  
- `I`: input current  
- `β`: decay factor (memory retention)

A larger β → longer-lasting membrane potential.

---

### 2️⃣ Fire (Thresholding)

\[
S = spike\_fn(U - \theta)
\]

- Output spike **1** if membrane potential exceeds threshold  
- Otherwise output **0**

The spike function uses a **Triangular Surrogate Gradient** during training to make the threshold differentiable.

---

### 3️⃣ Reset

\[
U = U - S \cdot \theta
\]

- If a spike occurs → subtract θ  
- If no spike → keep membrane potential

---

### Implementation-related features

Other functions such as:

- `register_buffer`
- `nn.Parameter`
- `torch.stack`

are used mainly for:

- Automatically moving parameters to GPU  
- Supporting both 5-dim (B,T,C,H,W) and 3-dim inputs  
- Combining spike outputs across timesteps  

---

## 6. Dimension Definitions: `[B, T, C, H, W]` and `[B, T, D]`

| Dimension | Meaning | Source |
|-----------|---------|--------|
| **B** | batch size | data loader |
| **T** | timesteps | SNN unfolding |
| **C** | channels | Conv output features |
| **H** | height | subcarriers |
| **W** | width | OFDM symbols |

- In the SNN convolution stage → **[B, T, C, H, W]**  
- In the final ANN layer → **[B, T, D]** (flattened features)

LIF operates **only** on temporal integration across timesteps.  
It does **not** create new spatial features or channels.

---

## 7. Key Concept Summary

| Key Idea | Explanation |
|----------|-------------|
| LIF does not extract features | Only convolution layers extract spatial/channel features |
| Trainable weights exist only in conv layers | LIF has no learnable parameters |
| β and θ are fixed hyperparameters | Paper uses β = 0.9, θ = 1.0 |
| Surrogate gradient is essential | Triangular surrogate stabilizes training |
| LIF controls temporal dynamics | Implements the Integrate–Fire–Reset loop |

---

## 8. Hierarchy Mapping (Top → Bottom)

| Level | Name | Description |
|--------|------|-------------|
| System | SpikingRx Receiver | SNN replaces traditional channel estimation and equalization |
| Network | SEW-ResNet Blocks + ANN Readout | Convolution → LIF → Residual → LLR |
| Module | LIF Neuron | Implements membrane dynamics |
| Code | `lif_neuron.py` | Pure neuron behavior, no convolution |
| Core Mechanism | Integrate–Fire–Reset | Fundamental neuron cycle |

---

## 9. Overall Conclusion

`lif_neuron.py` is the **spiking neuron module** of SpikingRx.

It performs only:

- Membrane potential integration  
- Thresholding + spike generation  
- Reset after firing  

All other helper logic (surrogate gradient, parameter buffers, dimension handling) exists to support training within PyTorch and integration into SEW-ResNet.

This module converts continuous convolution outputs into time-series **binary spikes (0/1)**, enabling the ANN readout to aggregate temporal information and generate LLRs for decoding.

