import numpy as np

def lstm_cell_backward(
    dh_t, dC_t, cache,
    W_f, W_i, W_C, W_o
):
    """
    Performs a single backward pass of an LSTM cell.

    Parameters
    ----------
    dh_t : np.ndarray, shape (dh,)
        Gradient w.r.t. current hidden state h_t
    dC_t : np.ndarray, shape (dh,)
        Gradient w.r.t. current cell state C_t
    cache : dict
        Values from forward pass
    W_f, W_i, W_C, W_o : np.ndarray, shape (dh, dh + dx)

    Returns
    -------
    dx_t : np.ndarray, shape (dx,)
    dh_prev : np.ndarray, shape (dh,)
    dC_prev : np.ndarray, shape (dh,)
    dW_f, dW_i, dW_C, dW_o : np.ndarray, shape (dh, dh + dx)
    db_f, db_i, db_C, db_o : np.ndarray, shape (dh,)
    """

    # --------------------------------------------------
    # Unpack cache
    # --------------------------------------------------
    f_t = cache["f_t"]
    i_t = cache["i_t"]
    C_tilde = cache["C_tilde"]
    C_t = cache["C_t"]
    o_t = cache["o_t"]
    tanh_C_t = cache["tanh_C_t"]
    h_prev = cache["h_prev"]
    C_prev = cache["C_prev"]
    x_t = cache["x_t"]
    concat = cache["concat"]

    # --------------------------------------------------
    # 1. Total gradient w.r.t. C_t
    # --------------------------------------------------
    dC_total = dC_t + dh_t * o_t * (1.0 - tanh_C_t**2)

    # --------------------------------------------------
    # 2. Gradients through cell update
    # --------------------------------------------------
    dC_prev = dC_total * f_t
    dC_tilde = dC_total * i_t
    df_t = dC_total * C_prev
    di_t = dC_total * C_tilde

    # --------------------------------------------------
    # 3. Output gate gradient
    # --------------------------------------------------
    do_t = dh_t * tanh_C_t

    # --------------------------------------------------
    # 4. Backprop through activations
    # --------------------------------------------------
    df_pre = df_t * f_t * (1.0 - f_t)
    di_pre = di_t * i_t * (1.0 - i_t)
    do_pre = do_t * o_t * (1.0 - o_t)
    dCtilde_pre = dC_tilde * (1.0 - C_tilde**2)

    # --------------------------------------------------
    # 5. Parameter gradients
    # --------------------------------------------------
    dW_f = np.outer(df_pre, concat)
    dW_i = np.outer(di_pre, concat)
    dW_o = np.outer(do_pre, concat)
    dW_C = np.outer(dCtilde_pre, concat)

    db_f = df_pre
    db_i = di_pre
    db_o = do_pre
    db_C = dCtilde_pre

    # --------------------------------------------------
    # 6. Gradient w.r.t. concatenated input
    # --------------------------------------------------
    dconcat = (
        W_f.T @ df_pre +
        W_i.T @ di_pre +
        W_o.T @ do_pre +
        W_C.T @ dCtilde_pre
    )

    dh_prev = dconcat[:h_prev.shape[0]]
    dx_t = dconcat[h_prev.shape[0]:]

    return (
        dx_t.astype(np.float32),
        dh_prev.astype(np.float32),

      
      #LSTM Cell Backward Pass
#Problem Description
#Backpropagation through an LSTM cell is more complex than a vanilla RNN because of the gating mechanism and the cell state. The gradients flow through multiple gates (forget, input, output) and the cell state, requiring careful application of the chain rule.

#In this problem, you will implement the backward pass of a single LSTM cell, computing gradients with respect to all inputs and parameters.
