import numpy as np

def minibatch_discrimination(features, T):
    """
    Minibatch Discrimination

    Inputs
    ------
    features : np.ndarray
        Shape (N, A)  -> batch_size x input_dim
    T : np.ndarray
        Shape (A, B, C) -> input_dim x num_kernels x kernel_dim

    Returns
    -------
    mb_features : np.ndarray
        Shape (N, B)
    """

    N, A = features.shape
    A_t, B, C = T.shape
    assert A == A_t, "Feature dimension mismatch"

    # ---------------------------------------------------
    # Step 1: Compute M = f(x) ⋅ T
    # ---------------------------------------------------
    # Reshape T from (A, B, C) -> (A, B*C)
    T_flat = T.reshape(A, B * C)

    # Matrix multiplication: (N, A) @ (A, B*C) -> (N, B*C)
    M_flat = features @ T_flat

    # Reshape to (N, B, C)
    M = M_flat.reshape(N, B, C)

    # ---------------------------------------------------
    # Step 2: Compute pairwise L1 distances
    # ---------------------------------------------------
    # Expand for broadcasting:
    # M_i -> (N, 1, B, C)
    # M_j -> (1, N, B, C)
    M_i = M[:, np.newaxis, :, :]
    M_j = M[np.newaxis, :, :, :]

    # L1 distance over kernel_dim C
    # Result shape: (N, N, B)
    dists = np.sum(np.abs(M_i - M_j), axis=3)

    # ---------------------------------------------------
    # Step 3: Exponentiate and sum over batch
    # ---------------------------------------------------
    # exp(-d_ij(b))
    exp_neg_dist = np.exp(-dists)

    # Sum over j dimension -> (N, B)
    mb_features = np.sum(exp_neg_dist, axis=1)

    # ---------------------------------------------------
    # Step 4: Remove self-comparison
    # ---------------------------------------------------
    mb_features -= 1.0

    return mb_features
