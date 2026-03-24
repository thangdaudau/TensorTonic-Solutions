import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    dL_dht = dh_next
    dht_dzt = 1 - h_t * h_t
    dzt_dwhh = h_prev
    dzt_dhprev = W_hh

    dL_dzt = dL_dht * dht_dzt
    dW_hh = dL_dzt.T @ dzt_dwhh
    dh_prev = dL_dzt @ dzt_dhprev
    return dh_prev, dW_hh