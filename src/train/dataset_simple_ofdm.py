import numpy as np
import torch
from torch.utils.data import Dataset

class SimpleOFDMDataset(Dataset):
    """
    QPSK OFDM → 32×32 → (T, C=2, H, W)
    """

    def __init__(self, num_samples=2000, H=32, W=32, T=5, snr_db=20):
        self.num_samples = num_samples
        self.H = H
        self.W = W
        self.T = T
        self.snr_db = snr_db

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # ---------------------------
        # 1) bits → QPSK
        # ---------------------------
        bits = np.random.randint(0, 2, (self.H, self.W, 2)).astype(np.float32)

        # Gray mapping:
        # 00 → +1 + j*1
        # 01 → -1 + j*1
        # 11 → -1 + j*-1
        # 10 → +1 + j*-1
        I = 2*bits[...,0] - 1
        Q = 2*bits[...,1] - 1
        symbols = I + 1j*Q

        # normalize power
        symbols = symbols / np.sqrt(2)

        # ---------------------------
        # 2) AWGN
        # ---------------------------
        snr_linear = 10 ** (self.snr_db / 10)
        noise_var = 1 / snr_linear
        noise = np.sqrt(noise_var/2) * (
            np.random.randn(self.H, self.W) +
            1j * np.random.randn(self.H, self.W)
        )

        rx = symbols + noise

        # ---------------------------
        # 3) produce (T, 2, H, W)
        # ---------------------------
        chw = np.stack([rx.real.astype(np.float32),
                        rx.imag.astype(np.float32)], axis=0)

        # repeat across T
        x = np.repeat(chw[np.newaxis, ...], self.T, axis=0)

        # to tensor
        x = torch.tensor(x, dtype=torch.float32)

        # label bits (float for BCE)
        y = torch.tensor(bits, dtype=torch.float32)

        return x, y


