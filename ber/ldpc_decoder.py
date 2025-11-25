import numpy as np

# 38.212 NR LDPC Base Graphs (BG1 and BG2)
# These are standard parity-check base matrices.
# 0 = no connection, positive entry = cyclic shift
from src.ber.nr_ldpc_basegraph import BG1, BG2   # we will create this file below


def ldpc_bp_decode(llr, H, Z, max_iter=25):
    """
    LDPC Belief Propagation Decoder (Min-Sum)
    llr  : soft LLR from SpikingRx, length = N bits
    H    : parity-check matrix (expanded by Z)
    Z    : lifting size
    """
    M, N = H.shape  # parity eqs, length
    llr = llr.copy()

    # initialize messages
    msg_vc = np.zeros((M, N))          # variable → check
    msg_cv = np.zeros((M, N))          # check → variable

    # neighbors
    checks_for_var = [np.where(H[:, i] != -1)[0] for i in range(N)]
    vars_for_check = [np.where(H[j, :] != -1)[0] for j in range(M)]

    for _ in range(max_iter):

        # check → variable
        for j in range(M):
            vs = vars_for_check[j]
            for v in vs:
                others = vs[vs != v]
                signs = np.prod(np.sign(msg_vc[j, others]))
                mins  = np.min(np.abs(msg_vc[j, others]))
                msg_cv[j, v] = signs * mins

        # variable → check
        for i in range(N):
            cs = checks_for_var[i]
            for c in cs:
                others = cs[cs != c]
                msg_vc[c, i] = llr[i] + np.sum(msg_cv[others, i])

        # tentative decode
        total_llr = llr + np.sum(msg_cv[:, :], axis=0)
        decoded = (total_llr < 0).astype(np.uint8)

        # stop if parity satisfied
        syndrome = (H.dot(decoded) % 2)
        if not syndrome.any():
            return decoded

    return decoded  # max iter reached


def nr_ldpc_decode(llr, BG, Z, Kb, F):
    """
    Wrapper for NR LDPC (38.212).
    llr: soft LLR
    BG : 1 or 2 (base graph)
    Z  : lifting size
    Kb : Kb from segmentation (number of info bits in CB)
    F  : filler bits
    """
    # determine correct base graph
    if BG == 1:
        H_base = BG1
    else:
        H_base = BG2

    # construct full expanded H
    M0, N0 = H_base.shape
    M = M0 * Z
    N = N0 * Z

    H_full = -np.ones((M, N), dtype=int)  # -1 means no edge

    for r in range(M0):
        for c in range(N0):
            if H_base[r, c] >= 0:
                shift = H_base[r, c]
                for z in range(Z):
                    H_full[r*Z + z, c*Z + (z + shift) % Z] = 1

    # decode
    bits = ldpc_bp_decode(llr, H_full, Z)

    # remove filler bits
    info = bits[:Kb - F]

    return info

