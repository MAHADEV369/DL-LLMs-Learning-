import numpy as np

def rnn_cell_forward(x_t, h_prev, W_x, W_h, b):
    """
    Forward pass for a single vanilla RNN cell
    """
    h_t = np.tanh(
        W_x @ x_t +
        W_h @ h_prev +
        b
    )
    return h_t.astype(np.float32)
