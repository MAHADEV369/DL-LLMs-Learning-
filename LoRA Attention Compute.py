import numpy as np

def lora_attention_compute(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q (np.ndarray): (batch_size, seq_len, d_model)
        K (np.ndarray): (batch_size, seq_len, d_model)
        V (np.ndarray): (batch_size, seq_len, d_model)

    Returns:
        np.ndarray: (batch_size, seq_len, d_model)
    """
    d_k = Q.shape[-1]

    # (batch, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Numerically stable softmax along last axis (keys)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # (batch, seq_len, d_model)
    output = np.matmul(attn_weights, V)

    return output
