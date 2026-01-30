import numpy as np

def lora_merge_multiple(adapters, weights):
    """
    Merge multiple LoRA adapters into a single update.

    Args:
        adapters (list): List of (A, B) tuples
            A: (out_dim, rank_i)
            B: (rank_i, in_dim)
        weights (list): List of float weights, same length as adapters

    Returns:
        np.ndarray: (out_dim, in_dim) merged LoRA update
    """
    assert len(adapters) == len(weights), "Adapters and weights must match in length"

    # Infer output shape from first adapter
    A0, B0 = adapters[0]
    out_dim, _ = A0.shape
    _, in_dim = B0.shape

    merged_update = np.zeros((out_dim, in_dim), dtype=A0.dtype)

    for (A, B), w in zip(adapters, weights):
        merged_update += w * (A @ B)

    return merged_update
å
