# 2025/11/06 — SpikingRx Simulation Environment Setup Log

## Purpose
Create an isolated, GPU-enabled Python development environment on Windows  
for simulating SpikingRx (including LIF, SEW-ResNet, and the main model).

This environment must:

- Not interfere with system Python  
- Support CUDA computation (NVIDIA GTX 950M)  
- Allow switching between CPU and GPU  
- Keep a clean and extensible structure  

---

## 1. Project Directory Structure

In CMD, run:

```bash
D:
mkdir D:\dev
mkdir D:\dev\spikingrx
cd /d D:\dev\spikingrx
mkdir src
mkdir data
mkdir notebooks
```

**Result:**

```
D:\dev\spikingrx
│
├── src\
├── data\
├── notebooks\
└── (later: .venv and activate_env.bat will be added)
```

**Purpose:**

- `src/` → neuron models, SpikingRx architecture, training scripts  
- `data/` → test or simulation data (OFDM, LLR, etc.)  
- `notebooks/` → training logs and visualization notes  

---

## 2. Create a Virtual Environment

Under `D:\dev\spikingrx`, run:

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

If successful, the CMD prompt becomes:

```
(.venv) D:\dev\spikingrx>
```

**Purpose:**  
All packages (PyTorch, NumPy, etc.) are installed inside `.venv`,  
so they do not affect other projects or the system Python.

---

## 3. Create an Auto-Activation Batch File

Create `activate_env.bat` with the following content:

```bat
@echo off
D:
cd D:\dev\spikingrx
call .venv\Scripts\activate
cmd
```

**Purpose:**  
By double-clicking this batch file, you can:

1. Automatically switch to drive D:  
2. Change directory to the project folder  
3. Activate the virtual environment  
4. Open a CMD window ready to use  

---

## 4. Install PyTorch with CUDA Support

Inside the virtual environment, run:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Then verify:

```python
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

**Actual result:**

```
Torch: 1.12.1+cu116
CUDA available: True
GPU: NVIDIA GeForce GTX 950M
```

This means PyTorch can successfully access the GPU.

---

## 5. Test GPU Computation Capability

In Python interactive mode:

```python
import torch
x = torch.randn(4096, 4096, device='cuda')
y = x @ x.t()
torch.cuda.synchronize()
print("Matmul OK. y shape:", y.shape)
```

**Result:**

```
Matmul OK. y shape: torch.Size([4096, 4096])
```

The GPU can perform large matrix multiplication correctly.

---

## 6. Fix NumPy Compatibility Warning

During the first run, the following warning appeared:

```
A module compiled using NumPy 1.x cannot be run in NumPy 2.x
```

This indicates that this PyTorch version does not support NumPy 2.x.

**Fix:**

```bash
pip install numpy==1.26.4
```

After downgrading NumPy, everything runs normally.

---

## 7. Concepts and Validation

| Concept              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Virtual environment `.venv` | Each project manages its own packages; environments do not interfere with each other. |
| CUDA verification    | Confirms that GPU driver, PyTorch CUDA build, and hardware are consistent. |
| CMD GPU test         | Quickly checks whether GPU matmul fits within 2 GB VRAM and runs correctly.|
| Isolated simulation  | SpikingRx simulation is developed separately from OAI, so they do not affect each other. |

---

## 8. Current Status and Next Steps

**Current:**

- Python 3.10.11 + CUDA environment is ready  
- Virtual environment and project structure are set up  
- GPU computation test has passed  

**Next steps:**

1. Create `src/models/lif_neuron.py` to implement the LIF firing model  
2. Create `src/tests/test_lif_forward.py` to test spike outputs  
3. Add Kaiming Normal initialization utilities  

---

## Key Notes

- The current GPU test is only to confirm that the environment supports CUDA;  
  it does **not** mean the implementation must always run on GPU.  
- In practice, you can freely switch the device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The program will automatically choose the available device (CPU/GPU).


