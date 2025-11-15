# 1115 SpikingRx System Architecture Overview
---

Objective:  
Use a “four-layer flowchart”, from system level → training flow → model forward → single SEW block, to clearly explain the role, data flow, and tensor dimensions of SpikingRx in the 5G PDSCH receiver chain.

(Note: The textual description follows the original architecture of the paper; implementation adjustments such as number of channels / hidden layers will be specified separately.)

---

# 0. Notation and Dimension Description (B / T / C / H / W)

All of the flowcharts in the following four layers will use the same set of dimension symbols:

B: batch size  
Number of OFDM frames processed simultaneously in one training / inference pass.

T: time steps of the SNN  
The same OFDM I/Q grid is fed into the LIF network T times, so that the membrane potential accumulates → spike train (in the paper, the input is copied T times and then fed into the SNN).

C: number of channels  
At the beginning, you can think of it as the two channels I/Q. After Conv / ResNet / SEW, it becomes higher-dimensional feature channels.

Paper example setting: Conv + 7 SEW ResNet blocks all use 128 channels.  
In this project implementation, we can adopt reduced configurations such as 16 / 32 / 64 (matching your flowchart).

H, W: 2D coordinates of the OFDM grid  
Corresponding to the number of OFDM symbols M and the number of subcarriers N in the paper:

H: OFDM symbol index (time domain)  
W: subcarrier index (frequency domain)

---

## 0.1 Core I/O of SpikingRx

From the perspective of the paper, the input is:

Time domain: after CP removal and FFT, we obtain the frequency-domain OFDM resource grid.

Two main input matrices:

Y ∈ C^(M×N): complete received OFDM resource grid (data + pilots)  
P′ ∈ C^(M×N): pilot grid (pilot values at pilot positions, zeros elsewhere)

After C2R (complex→real) + concatenation + replication T times along the time dimension:

It can be abstracted as a tensor:

[space for the tensor expression]

Here C can be regarded as “multiple real-valued channels”; in implementation it is often constructed using I/Q plus pilot-related information.

Output of SpikingRx to the LDPC decoder:

Each RE corresponds to B_t bits for the modulation:

Paper Table II setting: 16-QAM → B_t = 4 bits/RE  
If we use QPSK as a simplified example, then B_t = 2 bits/RE.

The final logits a_{b,h,w,l} themselves are exactly the LLRs of each bit:

[space for the equation]

Output tensor shape (I use H/W instead of M′/N in the paper):

[space for the tensor shape]

---

# 1. First Layer: System-Level Receiver Architecture

(Paper reference: Fig.1 traditional receiver, Fig.4 overall SpikingRx system position)  
(Code: SpikingRx main model)  
https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/spikingrx_model.py

Problem:  
In the entire 5G PDSCH receiver chain, which part is replaced by SpikingRx? What are the inputs / outputs?

---

## 1.1 Flowchart (System Level)

[space for system-level flowchart]

(Similar to Fig.4 in the paper.)

Overall structure (right → left):

Transmitter (same as in Sec.II of the paper)

Information bits → TB encoding (LDPC + rate matching + scrambling)  
Bits → QAM symbols (the paper uses 16-QAM)  
Data + DMRS mapped to PRBs → forming a resource grid  
IFFT → add CP → channel

Front half of the receiver (traditional 5G)

CP removal  
FFT: perform FFT on each OFDM symbol to obtain the frequency-domain resource grid Y

SpikingRx Receiver (new component of the paper)

Position: placed after “CP removal + FFT”

Functions replaced:

channel estimation (LS + interpolation)  
LMMSE equalization  
symbol de-mapping and LLR computation

Input: the entire received OFDM resource grid Y and pilot grid P′, after C2R + concatenation + replication T times, forming a tensor x(B, T, C, H, W).

Output: bit-wise LLR for each RE: LLR(B, H, W, B_t).

Back-end Transport Block Decoder

LLR → (soft-demapping already done)  
Directly feed into rate de-matcher + LDPC decoder → TB decoding and CRC checking.

---

## 1.2 Summary of Data Shapes in Layer 1

FFT → SpikingRx:

[space for the corresponding tensor expression]

SpikingRx → TB Decoder:

[space for the corresponding tensor expression]

---

# 2. Second Layer: Flow of Training One Epoch (Forward / Backward / Update)

(Paper reference: Sec. III-C Training Method: Forward → Loss → Backprop)

Problem:  
When training SpikingRx, what steps does each batch go through within one epoch?

---

## 2.1 Flowchart (Epoch)

[space for epoch flowchart]

Using the BCE loss in Sec.III-C of the paper as the basis, one epoch can be broken down as:

Take one training batch

Receiver input:

[space for the input tensor expression]

Label: the true bits of each RE:

[space for the label tensor expression]

Forward Pass: SpikingRx(x; θ)

Using the current weights θ, perform one complete forward pass:

LLR = SpikingRx(x; θ)

Shape: LLR ∈ R^(B×H×W×B_t).

Loss computation (BCE, multi-label)

The loss in the paper is:

[space for the loss equation]

In implementation, we can directly regard the logits as LLRs and compute the loss via BCEWithLogitsLoss.

Backward Pass (including surrogate gradient)

For ANN layers: standard backprop.  
For SNN (LIF) parts:

Forward uses LIF update:

U[t] = βU[t−1] + W X[t] − S_out[t−1] θ

Backward: approximate the derivative of the Heaviside function ∂S/∂U with a continuous function (the paper uses a threshold-shifted sigmoid).

Optimizer Update (Adam-W)

Use Adam-W to update all weights:

W ← W − η * ΔW

After finishing one batch → proceed to the next batch, until the entire dataset has been iterated once = 1 epoch.

# 3. Third Layer: SpikingRx Forward Pipeline (Internal Data Flow of the Model)

(Paper reference: Fig.3(e) Architecture, Table I Hyperparameters)

Problem:  
Ignoring training and looking only at inference, how does one x(B, T, C, H, W) become LLR(B, H, W, B_t)?

---

## 3.1 Flowchart (Forward Pipeline)

[space for the forward-pipeline diagrams]

According to Fig.3(e) and Table I of the paper, the data flow can be written as:

OFDM Grid Input

In batched form:

[space for the input tensor shape]

C2R & Concatenation (preprocessing in the paper)

Convert the complex Y, P′ into real channels, and replicate them T times to form the SNN time dimension.  
This part in the code can be implemented either in the data loader or at the very beginning of SpikingRx.forward().

Initial Conv2D + LIF (StemConv)

Table I: Conv2D with 128 filters, kernel 3×3, followed by a LIF neuron (Initial LIF).  
In your flowchart, this can be drawn as “2D Conv + SpikeNorm + LIF”.

Output:

[space for the tensor shape]

In the paper: C₀ = 128.  
In a reduced implementation you can use C₀ = 16.

Seven consecutive SEW / ResNet Blocks

Table I: Traditional / SEW ResNet-1 to ResNet-7, all with 128 filters and 3×3 kernels.

In your diagram, you can name them “Block1–Block6” or “Block1–Block7”.

If your implementation uses a channel schedule (16→16→32→32→64→64→64), then the input and output of each layer are:

[space for the list of shapes, e.g. (B,T,C,H,W) at each stage]

Temporal Mean over T (spike rate over time)

From the time-domain output Y_{b,t,c,h,w} of the last SEW block, compute the spike rate:

[space for rate equation]

Result:

[space for resulting tensor shape]

Readout ANN / Conv Head → logits / LLR

In Table I of the paper:

Conv2D: filters = B_t, kernel = 1×1  
Sigmoid: obtain the soft probability p_e(b,l,m′,n | y) at each time step  
Mean over T: average the T soft outputs (or directly use the logits as LLR)

In your current implementation:

Conv1×1: C_last → 32 → ReLU  
Conv1×1: 32 → B_t (for example, 2 for QPSK)

Finally arranged as:

[space for LLR tensor shape]

---

# 4. Fourth Layer: Internal Structure of a Single SEW Block

(Paper reference: Fig.5(b) SEW-ResNet Block)  
(Code: SEW / LIF / Norm / Conv)

SEW Block:  
https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/sew_block.py

LIF neuron:  
https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/lif_neuron.py

Norm layer:  
https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/norm_layer.py

Conv block:  
https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/conv_block.py

Problem:  
Within one SEW block, how are “Conv / Norm / LIF / Shortcut” connected mathematically?  
What is the difference compared to a traditional ResNet block?

---

## 4.1 Flowchart (SEW Block)

[space for SEW block diagram]

(Similar to Fig.5(b) in the paper.)

First, look at a single time step t, with both input and output in R^(B×C×H×W).

Input: x_t ∈ R^(B×C_in×H×W)  
This is the slice at time step t, coming from the previous block or from the output of the Initial Conv+LIF.

Conv2D (main path)

kernel: 3×3, padding = 1  
channels: C_in → C_out (most blocks in the paper use 128→128)

We obtain:

[space for the intermediate tensor]

Normalization

Channel-wise normalization (SpikeNorm / BatchNorm, etc.)  
The purpose is to keep the LIF input within a suitable range for spiking.

LIF Neuron

For each (b, c, h, w), maintain a membrane potential U[t]:

[space for membrane update equation]

Produce spikes:

[space for spike generation equation]

Resulting spike map:

[space for spike tensor]

Shortcut branch (identity / 1×1 Conv)

If C_in = C_out:

The shortcut is identity: s_c^t = x_t.

If the number of channels is different:

Use a 1×1 Conv to align the dimensions:  
s_c^t = Conv1×1(x_t)

SEW “input–output combination method”

Traditional ResNet: simply sums, g = I + O.

SEW ResNet: can choose different element-wise operations (AND, IAND, ADD, etc.):

g = I + O (ADD)  
g = I ∧ O (AND)  
g = (1 − I) ∧ O

In the SpikingRx implementation, the paper performs ablation studies over different combinations and selects the ones with better performance / energy efficiency.

Final output

Using ADD as an example:

y_t = s_t + s_c^t

Collect all time steps:

Y ∈ R^(B×T×C_out×H×W)
