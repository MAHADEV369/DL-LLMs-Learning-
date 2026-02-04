import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lstm_block_forward(
    x_indices, E,
    h_init, C_init,
    W_f, W_i, W_C, W_o,
    b_f, b_i, b_C, b_o,
    W_out, b_out
):
    """
    Full LSTM block forward pass:
    Embedding -> LSTM sequence -> Linear output
    """

    # -------------------------
    # 1. Embedding Lookup
    # -------------------------
    X = E[x_indices]          # (T, d_emb)
    T = X.shape[0]
    d_h = h_init.shape[0]

    # -------------------------
    # 2. LSTM Forward Sequence
    # -------------------------
    H = np.zeros((T, d_h), dtype=np.float32)

    h_t = h_init
    C_t = C_init

    for t in range(T):
        x_t = X[t]                                # (d_emb,)
        concat = np.concatenate([h_t, x_t])      # (d_h + d_emb,)

        f_t = sigmoid(W_f @ concat + b_f)
        i_t = sigmoid(W_i @ concat + b_i)
        C_hat_t = np.tanh(W_C @ concat + b_C)
        o_t = sigmoid(W_o @ concat + b_o)

        C_t = f_t * C_t + i_t * C_hat_t
        h_t = o_t * np.tanh(C_t)

        H[t] = h_t

    # -------------------------
    # 3. Linear Output Layer
    # -------------------------
    Y = H @ W_out.T + b_out     # (T, d_out)

    return Y.astype(np.float32)




#LSTM Full Block
#Problem Description
#This is the final integration task for the LSTM series. You will combine the components you've built (embedding lookup, LSTM sequence forward pass, and linear output layer) to perform a forward pass through a complete LSTM block.

#The goal is to wire up the components you've implemented in previous tasks into a complete LSTM forward pass: embedding → LSTM sequence → linear output.


