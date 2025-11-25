import numpy as np

def awgn_channel(x, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_var) * np.random.randn(*x.shape)
    return x + noise

