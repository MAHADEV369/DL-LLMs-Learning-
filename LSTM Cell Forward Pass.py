import numpy as np

def sigmoid(x):
    """
    Numerically stable sigmoid function.
    """
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def lstm_cell_forward(
    x_t, h_prev, C_prev,
    W_f, W_i, W_C, W_o,
    b_f, b_i, b_C, b_o
):
    """
    Performs a single forward pass of an LSTM cell.

    Parameters
    ----------
    x_t : np.ndarray, shape (dx,)
    h_prev : np.ndarray, shape (dh,)
    C_prev : np.ndarray, shape (dh,)
    W_f, W_i, W_C, W_o : np.ndarray, shape (dh, dh + dx)
    b_f, b_i, b_C, b_o : np.ndarray, shape (dh,)

    Returns
    -------
    h_t : np.ndarray, shape (dh,)
    C_t : np.ndarray, shape (dh,)
    cache : dict (for backward pass)
    """

    # ------------------------------------------------
    # 1. Concatenate previous hidden state and input
    # ------------------------------------------------
    concat = np.concatenate([h_prev, x_t], axis=0)  # (dh + dx,)

    # ----------------
    # 2. Forget gate
    # ----------------
    f_t = sigmoid(W_f @ concat + b_f)  # (dh,)

    # ----------------
    # 3. Input gate
    # ----------------
    i_t = sigmoid(W_i @ concat + b_i)  # (dh,)

    # --------------------------------
    # 4. Candidate cell state
    # --------------------------------
    C_tilde = np.tanh(W_C @ concat + b_C)  # (dh,)

    # -----------------------------
    # 5. Cell state update
    # -----------------------------
    C_t = f_t * C_prev + i_t * C_tilde  # (dh,)

    # ----------------
    # 6. Output gate
    # ----------------
    o_t = sigmoid(W_o @ concat + b_o)  # (dh,)

    # --------------------
    # 7. Hidden state
    # --------------------
    tanh_C_t = np.tanh(C_t)
    h_t = o_t * tanh_C_t  # (dh,)

    # --------------------
    # 8. Cache for backward
    # --------------------
    cache = {
        "f_t": f_t,
        "i_t": i_t,
        "C_tilde": C_tilde,
        "C_t": C_t,
        "o_t": o_t,
        "tanh_C_t": tanh_C_t,
        "h_prev": h_prev,
        "C_prev": C_prev,
        "x_t": x_t,
        "concat": concat
    }

    return h_t.astype(np.float32), C_t.astype(np.float32), cache





#LSTM Cell Forward Pass
#Problem Description
#Long Short-Term Memory (LSTM) networks are a type of recurrent neural network designed to address the vanishing gradient problem in vanilla RNNs. LSTMs use a gating mechanism to control the flow of information through the cell state, allowing them to learn long-term dependencies.

#At the core of an LSTM is the LSTM cell, which processes one time step of input and updates both a hidden state and a cell state. In this problem, you will implement the forward pass of a single LSTM cell from scratch.
