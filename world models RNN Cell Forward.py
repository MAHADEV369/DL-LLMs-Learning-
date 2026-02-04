import numpy as np

def rnn_cell_forward(x_t, h_prev, W_x, W_h, b):
    """
    Compute the forward pass of a vanilla RNN cell.

    Parameters:
    x_t : np.ndarray of shape (d_x,), dtype float32
        Input vector at time t
    h_prev : np.ndarray of shape (d_h,), dtype float32
        Previous hidden state
    W_x : np.ndarray of shape (d_h, d_x), dtype float32
        Input weight matrix
    W_h : np.ndarray of shape (d_h, d_h), dtype float32
        Hidden state weight matrix
    b : np.ndarray of shape (d_h,), dtype float32
        Bias vector

    Returns:
    h_t : np.ndarray of shape (d_h,), dtype float32
        New hidden state
    """
    z = np.dot(W_x, x_t) + np.dot(W_h, h_prev) + b
    h_t = np.tanh(z)
    return h_t
