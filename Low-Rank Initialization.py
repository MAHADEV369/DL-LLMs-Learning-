import numpy as np

def low_rank_init(out_dim, in_dim, rank, seed=42):
    """
    Initialize LoRA low-rank adapter matrices.

    Parameters
    ----------
    out_dim : int
        Output dimension of the layer
    in_dim : int
        Input dimension of the layer
    rank : int
        Rank of the low-rank decomposition
    seed : int, optional
        Random seed for deterministic initialization (default: 42)

    Returns
    -------
    A : np.ndarray
        Shape (out_dim, rank), float32, initialized to zeros
    B : np.ndarray
        Shape (rank, in_dim), float32, initialized from N(0, 0.01)
    """
    rng = np.random.RandomState(seed)

    # A initialized to zeros
    A = np.zeros((out_dim, rank), dtype=np.float32)

    # B initialized with small random values
    B = rng.normal(loc=0.0, scale=0.01, size=(rank, in_dim)).astype(np.float32)

    return A, B
#Problem Description
#LoRA (Low-Rank Adaptation) introduces trainable adapter matrices A and B to efficiently adapt large models. The adapter matrices are initialized with a specific pattern to ensure stable training.

#In this problem, you will initialize the LoRA adapter matrices A and B with a deterministic initialization scheme.

