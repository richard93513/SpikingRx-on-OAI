# 1115 SpikingRx System Architecture Summary

Objective:  
Use a “four-layer flowchart” from system level → training flow → model forward → single SEW block to clearly explain the role, data flow, and tensor dimensions of SpikingRx in the 5G PDSCH receiver chain.

(Note: The textual description mainly follows the original architecture in the paper; implementation details such as number of channels / hidden layers will be specified separately.)

---

# 0. Notation and Dimension Description (B / T / C / H / W)

All four layers of flowcharts will use the same dimension notation:

- **B: batch size**  
  Number of OFDM frames processed simultaneously in a single training / inference pass.

- **T: SNN time steps**  
  The same OFDM I/Q grid is repeatedly fed into the LIF network T times so that the membrane potential accumulates → spike train (in the paper, the input is copied T times and then fed into the SNN).

- **C: number of channels (channel)**  
  - At the beginning, you can regard it as two channels: I/Q.  
  - After Conv / ResNet / SEW, it becomes higher-dimensional feature channels.  
  - Paper example setting: Conv + 7 SEW ResNet blocks all use 128 channels.  
  - This project implementation can adopt reduced settings such as 16 / 32 / 64 (matching your flowchart).

- **H, W: 2D OFDM grid coordinates**  
  Corresponding to the number of OFDM symbols M and the number of subcarriers N in the paper:
  - **H:** OFDM symbol index (time domain)  
  - **W:** subcarrier index (frequency domain)

---

## 0.1 Core I/O of SpikingRx

From the paper’s viewpoint, the input is:

- **In the time-domain chain:**  
  after CP removal + FFT, we obtain the frequency-domain OFDM resource grid.

- **Two main complex-valued input matrices:**
  - Y ∈ C^(M×N): full received OFDM resource grid (data + pilots)  
  - P′ ∈ C^(M×N): pilot grid (pilot value at pilot locations, zero elsewhere)

After C2R (complex → real) + concatenation + replication along the time dimension T, this can be abstracted as a tensor:

- x ∈ ℝ^(B × T × C × H × W)

Here, C can be seen as “multiple real-valued channels”; in implementation it is usually constructed from I/Q together with pilot-related information.

---

### 0.1.1 Output of SpikingRx to the LDPC Decoder

- For each RE, there are B_t modulation bits:
  - Paper Table II setting: 16-QAM → B_t = 4 bits/RE  
  - Simplified example: QPSK → B_t = 2 bits/RE

- The final logits a_{b,h,w,l} themselves are the LLRs of each bit.

- Output tensor shape (using H/W instead of M′/N in the paper):
  - LLR ∈ ℝ^(B × H × W × B_t)

---

# 1. First Layer: System-Level Receiver Architecture

(Paper references: Fig.1 traditional receiver, Fig.4 overall SpikingRx system position)  
(Code: SpikingRx main model)  
https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/spikingrx_model.py

**Question:**  
In the entire 5G PDSCH receiver chain, which part is replaced by SpikingRx? What are the inputs / outputs?

---

## 1.1 Flowchart (System Level)

<img width="837" height="124" alt="image" src="https://github.com/user-attachments/assets/ee5b4d30-82aa-4734-b3b1-56b7b7f6efac" />

Overall structure (from right → left):

- **Transmitter (same as Sec.II in the paper)**  
  - Information bits → TB encoding (LDPC + rate matching + scrambling)  
  - Bits → QAM symbols (paper: 16-QAM)  
  - Data + DMRS are mapped onto PRBs → forming the resource grid  
  - IFFT → add CP → channel

- **Receiver front-end (traditional 5G part)**  
  - Remove CP  
  - FFT: perform FFT on each OFDM symbol to obtain the frequency-domain resource grid Y

- **SpikingRx Receiver (new part in the paper)**  
  - **Position:** inserted after “CP removal + FFT”  
  - **Functions it replaces:**  
    - channel estimation (LS + interpolation)  
    - LMMSE equalization  
    - symbol de-mapping & LLR computation  
  - **Input:**  
    - the full received OFDM resource grid Y  
    - the pilot grid P′  
    - after C2R + concatenation + replication along time T, this becomes the tensor  
      - x ∈ ℝ^(B × T × C × H × W)  
  - **Output:**  
    - bit-wise LLR for each RE  
    - LLR ∈ ℝ^(B × H × W × B_t)

- **Back-end Transport Block Decoder**  
  - takes the LLRs (soft demapping is already done)  
  - directly feeds them into rate de-matcher + LDPC decoder  
  - outputs decoded TB and performs CRC check

---

## 1.2 Summary of Data Shapes in Layer 1

- From FFT to SpikingRx:

  - x ∈ ℝ^(B × T × C × H × W)

- From SpikingRx to TB decoder:

  - LLR ∈ ℝ^(B × H × W × B_t)

---

# 2. Second Layer: One Epoch of Training (Forward / Backward / Update)

(Paper reference: Sec. III-C Training Method: Forward → Loss → Backprop)

**Question:**  
When training SpikingRx, within a single epoch, what steps are performed on each batch?

---

## 2.1 Flowchart (Epoch)

<img width="1213" height="66" alt="image" src="https://github.com/user-attachments/assets/3667928e-db6b-488b-a812-0b235b3f99f2" />

Using the BCE loss described in Sec.III-C of the paper, we can break one epoch into:

- **(1) Take one training batch**
  - Receiver input:
    - x ∈ ℝ^(B × T × C × H × W)
  - Ground-truth labels:
    - b ∈ {0,1}^(B × H × W × B_t)
    - the true bits of each RE

- **(2) Forward pass: SpikingRx(x; θ)**
  - Using the current parameters θ, perform one full forward pass:
    - LLR = SpikingRx(x; θ)
  - Shape:
    - LLR ∈ ℝ^(B × H × W × B_t)

- **(3) Loss computation (BCE, multi-label)**  
  - The loss used in the paper is binary cross-entropy over all bits:  
    - Loss = BCE(LLR, b)  (LLR as logits, b as targets)  
  - In implementation, we typically regard the logits as LLRs and use `BCEWithLogitsLoss`.

- **(4) Backward pass (including surrogate gradient for spikes)**  
  - For ANN layers:
    - use standard backpropagation.  
  - For SNN (LIF) layers:
    - forward membrane update:
      - U[t] = β U[t−1] + W X[t] − S_out[t−1] θ
    - the Heaviside derivative ∂S/∂U is replaced by a continuous surrogate function
      (the paper uses a threshold-shifted sigmoid).

- **(5) Optimizer update (Adam-W)**  
  - Use Adam-W to update all weights:
    - W ← W − η · ΔW

- After finishing one batch, proceed to the next batch until the entire dataset is processed once → that is 1 epoch.

# 3. Third Layer: SpikingRx Forward Pipeline (Internal Data Flow)

(Paper references: Fig.3(e) Architecture, Table I Hyperparameters)

**Question:**  
If we only look at inference (ignoring training), how does one tensor x(B, T, C, H, W) become LLR(B, H, W, B_t)?

---

## 3.1 Flowchart (Forward Pipeline)

<img width="865" height="102" alt="image" src="https://github.com/user-attachments/assets/bdd65378-0dfb-4b78-9640-504c074f2f8c" /><img width="780" height="91" alt="image" src="https://github.com/user-attachments/assets/052e7f7f-4e23-4ec6-9a5a-33926c4119cf" />

According to Fig.3(e) and Table I in the paper, we can describe the forward data flow as follows.

---

### 3.1.1 OFDM Grid Input

- Batched input tensor:
  - x ∈ ℝ^(B × T × C × H × W)
- Here H/W correspond to the OFDM symbol index and subcarrier index, C is the number of real-valued channels.

---

### 3.1.2 C2R and Concatenation (Pre-processing in the Paper)

- Convert the complex-valued Y and P′ into real channels.  
- Concatenate them along the channel dimension.  
- Replicate this along the time dimension T to create the SNN temporal axis.  

This pre-processing can be implemented either:

- inside the data loader, or  
- at the beginning of `SpikingRx.forward()`.

---

### 3.1.3 Initial Conv2D + LIF (StemConv)

- In Table I of the paper:
  - Conv2D:
    - number of filters: 128
    - kernel size: 3×3
  - followed by one LIF neuron (Initial LIF).

- In your flowchart, this can be drawn as:
  - “2D Conv + SpikeNorm + LIF”

- Output tensor shape:
  - x₀ ∈ ℝ^(B × T × C₀ × H × W)

- In the paper:
  - C₀ = 128  
- In a reduced implementation for this project:
  - C₀ can be set to 16.

---

### 3.1.4 Seven Consecutive SEW / ResNet Blocks

- In Table I:
  - Trad./SEW ResNet-1 to ResNet-7,
  - all with 128 filters and 3×3 kernels.

- In your own implementation and flowchart, you may denote them as:
  - Block1, Block2, …, Block6 or Block7.

- If your implementation uses a step-wise channel configuration such as:
  - 16 → 16 → 32 → 32 → 64 → 64 → 64

  then for each block k we have:

  - Input:
    - X_k ∈ ℝ^(B × T × C_in^k × H × W)
  - Output:
    - X_{k+1} ∈ ℝ^(B × T × C_out^k × H × W)

  where C_in^k and C_out^k follow the above sequence.

---

### 3.1.5 Temporal Mean over T (Spike Rate)

- From the time-resolved output of the last SEW block:
  - Y_{b,t,c,h,w}

- Compute the spike rate over the T time steps for each (b, c, h, w):

  - Rate[b,c,h,w] = (1 / T) · Σ_{t=1}^{T} S[b,t,c,h,w]

- Resulting tensor:

  - Rate ∈ ℝ^(B × C_last × H × W)

---

### 3.1.6 Readout ANN / Conv Head → Logits / LLR

- In Table I of the paper:

  - 1×1 Conv:
    - number of filters: B_t
    - kernel size: 1×1  

  - Sigmoid:
    - gives a soft probability p_e(b,l,m′,n | y) at each time step  

  - Temporal averaging over T:
    - either on the probabilities or in the logit domain,
    - finally producing bit-wise LLRs per RE.

- In your current implementation:

  - First 1×1 Conv:
    - C_last → 32
    - followed by ReLU  

  - Second 1×1 Conv:
    - 32 → B_t (for example B_t = 2 for QPSK)

- Final LLR tensor arrangement:

  - LLR ∈ ℝ^(B × H × W × B_t)

---

# 4. Fourth Layer: Internal Structure of a Single SEW Block

(Paper reference: Fig.5(b) SEW-ResNet Block)  
(Code: SEW / LIF / Norm / Conv)

- SEW block implementation:  
  - https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/sew_block.py  

- LIF neuron:  
  - https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/lif_neuron.py  

- Normalization layer:  
  - https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/norm_layer.py  

- Convolution block:  
  - https://github.com/richard93513/SpikingRx-on-OAI/blob/main/src/models/conv_block.py  

**Question:**  
Inside a single SEW block, how are “Conv / Norm / LIF / Shortcut” connected mathematically?  
What is the difference compared with a conventional ResNet block?

---

## 4.1 Flowchart (SEW Block)

<img width="1198" height="82" alt="image" src="https://github.com/user-attachments/assets/933665d1-2dda-477b-94f1-13813cb044a4" />

(Similar to Fig.5(b) in the paper.)

First, consider a single time step t. The input and output are defined on ℝ^(B × C × H × W).

---

### 4.1.1 Input

- x_t ∈ ℝ^(B × C_in × H × W)  
  - coming from the previous block or from the Initial Conv+LIF output,  
  - this is the slice at time step t.

---

### 4.1.2 Conv2D (Main Path)

- Convolution on the main branch:
  - kernel size: 3×3
  - padding: 1
  - channels: C_in → C_out
  - in most blocks of the paper: 128 → 128

- We obtain the intermediate feature map:

  - z_t ∈ ℝ^(B × C_out × H × W)

---

### 4.1.3 Normalization

- Apply channel-wise normalization (SpikeNorm / BatchNorm, etc.):

  - ẑ_t = Norm(z_t)

- Purpose:
  - keep the input to the LIF neurons within a suitable range so that spiking behaviour is neither saturated nor completely silent.

---

### 4.1.4 LIF Neuron

- For each spatial position (b, c, h, w), maintain a membrane potential U[t]:

  - U[t] = β · U[t−1] + ẑ_t − S_out[t−1] · θ

  where:
  - β is the leak factor,
  - θ is the firing threshold,
  - S_out[t−1] is the spike output at the previous time step.

- Generate spikes:

  - S_out[t] = H(U[t] − θ)

  (H is the Heaviside step function; in backprop it is replaced by a surrogate derivative.)

- The spike output map is:

  - S_out[t] ∈ ℝ^(B × C_out × H × W)

---

### 4.1.5 Shortcut Branch (Identity / 1×1 Conv)

- If C_in = C_out:
  - shortcut s_t is just identity:
    - s_t = x_t

- If C_in ≠ C_out:
  - use a 1×1 Conv to match dimensions:
    - s_t = Conv1×1(x_t)

---

### 4.1.6 SEW Element-Wise Merge

- Traditional ResNet:
  - simply y_t = s_t + S_out[t] (pure addition).

- SEW ResNet:
  - uses different element-wise operations between “input branch” I and “output branch” O:
    - ADD:  g = I + O  
    - AND:  g = I ∧ O  
    - IAND: g = (1 − I) ∧ O  

- In the SpikingRx implementation, the paper performs ablation over different SEW modes and selects the combination providing better performance / energy efficiency.

- In the simple ADD case:

  - y_t = s_t + S_out[t]

---

### 4.1.7 Final Output of One SEW Block

- For a single time step t:

  - y_t ∈ ℝ^(B × C_out × H × W)

- Collecting all time steps together, the block output is:

  - Y ∈ ℝ^(B × T × C_out × H × W)
