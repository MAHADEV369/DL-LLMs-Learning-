import numpy as np

def lora_apply_to_linear_layer(W, b, A, B, X, alpha):
    """
    Apply LoRA adaptation to a linear layer.

    Args:
        W (np.ndarray): Base weight matrix (out_dim, in_dim)
        b (np.ndarray): Bias vector (out_dim,)
        A (np.ndarray): LoRA A matrix (out_dim, rank)
        B (np.ndarray): LoRA B matrix (rank, in_dim)
        X (np.ndarray): Input (batch_size, in_dim) or (in_dim,)
        alpha (float): Scaling factor

    Returns:
        np.ndarray: Output (batch_size, out_dim) or (out_dim,)
    """

    # Compute LoRA weight update
    delta_W = alpha * (A @ B)          # (out_dim, in_dim)
    W_adapted = W + delta_W            # (out_dim, in_dim)

    # Unbatched input
    if X.ndim == 1:
        return W_adapted @ X + b       # (out_dim,)

    # Batched input
    else:
        return X @ W_adapted.T + b     # (batch_size, out_dim)


#Problem Description
#This problem combines all the LoRA components to apply LoRA adaptation to a complete linear layer. Given a base linear layer (weight W and bias b), LoRA adapters (A, B), and input X, compute the final output with the adapted weights.
