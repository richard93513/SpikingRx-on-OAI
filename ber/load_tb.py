import numpy as np

def load_tb_from_oai(path):
    """Load pre-LDPC TB bits dumped by OAI (uint8 raw)."""
    data = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(data)
    return bits

