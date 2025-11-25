import numpy as np
import torch
from src.ber.load_tb import load_tb_from_oai
from src.ber.simple_channel import awgn_channel
from src.models.spikingrx_model import SpikingRxModel

def run_pipeline(tb_path, snr_db, ckpt_path):
    # 1. load TB bits
    tb_bits = load_tb_from_oai(tb_path)
    print("Loaded TB bits:", tb_bits.shape)

    # 2. channel
    tx_symbols = 1 - 2 * tb_bits  # BPSK mapping (0→+1, 1→−1)
    rx_symbols = awgn_channel(tx_symbols, snr_db)

    # 3. convert to SpikingRx input format 變成 32×32 (TODO: 之後我幫你補完整)
    # placeholder:
    x = torch.zeros(1, 5, 2, 32, 32)

    # 4. load model
    model = SpikingRxModel()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    # 5. forward
    model.eval()
    with torch.no_grad():
        llr = model(x)

    print("Got LLR shape:", llr.shape)

    # 6. LDPC decode（之後我幫你補）
    # decoded_bits = ldpc_decode(llr)

    # 7. 計算 BER（之後補）
    # ber = np.mean(tb_bits != decoded_bits)

    print("Pipeline done.")

