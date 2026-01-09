import numpy as np

def lora_qkv_projection(
    X,
    Wq, Aq, Bq,
    Wk, Ak, Bk,
    Wv, Av, Bv,
    alpha
):
    """
    LoRA-adapted Q/K/V projection (NumPy only)

    X  : (B*S, D) or (B, S, D)
    W* : (D, D)
    A* : (D, r)
    B* : (r, D)
    alpha : float
    """

    original_shape = X.shape
    is_3d = (X.ndim == 3)

    # Flatten if needed: (B, S, D) → (B*S, D)
    if is_3d:
        B, S, D = X.shape
        X_flat = X.reshape(-1, D)
    else:
        X_flat = X
        D = X.shape[-1]

    # Compute adapted weights
    Wq_hat = Wq + alpha * (Aq @ Bq)
    Wk_hat = Wk + alpha * (Ak @ Bk)
    Wv_hat = Wv + alpha * (Av @ Bv)

    # Linear projections
    Q = X_flat @ Wq_hat.T
    K = X_flat @ Wk_hat.T
    V = X_flat @ Wv_hat.T

    # Restore original shape if needed
    if is_3d:
        Q = Q.reshape(B, S, D)
        K = K.reshape(B, S, D)
        V = V.reshape(B, S, D)

    return Q, K, V


#In transformer attention layers, LoRA can be applied to the query (Q), key (K), and value (V) projections separately. This problem computes the adapted Q, K, V projections by combining base weights with their respective LoRA adapters.

#For each projection (Q, K, V), we compute:

#Adapted projection = base_weight + alpha * (A @ B)
