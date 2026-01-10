import numpy as np

def effective_rank(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute effective rank contribution for LoRA matrices A and B.

    Parameters:
    A : np.ndarray of shape (out_dim, rank)
    B : np.ndarray of shape (rank, in_dim)

    Returns:
    float : effective rank contribution
    """
    # Extract rank
    rank = A.shape[1]

    # Frobenius norms
    norm_A = np.linalg.norm(A, ord='fro')
    norm_B = np.linalg.norm(B, ord='fro')

    # Matrix product
    AB = A @ B
    norm_AB = np.linalg.norm(AB, ord='fro')

    # Handle zero norm edge case
    if norm_AB == 0.0:
        return 0.0

    return rank * (norm_A * norm_B) / norm_AB
#Effective Rank
#Problem Description
#The effective rank of a matrix product A @ B provides insight into the information capacity of the low-rank decomposition. One common measure is the trace-based effective rank, which considers the singular values of the composed matrix.

#For this problem, we'll compute a simpler metric: the effective rank contribution as the ratio of the trace of (A @ B) to its Frobenius norm, which gives a sense of how "spread out" the adaptation is.
