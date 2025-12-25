import numpy as np

def class_token_prepend(patch_embeddings: np.ndarray,
                         class_token: np.ndarray) -> np.ndarray:
    """
    Prepends a learnable class token to patch embeddings.

    Args:
        patch_embeddings (np.ndarray): Shape (N, D), dtype float32
        class_token (np.ndarray): Shape (1, D), dtype float32

    Returns:
        np.ndarray: Shape (N+1, D), dtype float32
    """
    # --- Safety checks ---
    if patch_embeddings.ndim != 2 or class_token.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")

    if class_token.shape[0] != 1:
        raise ValueError("class_token must have shape (1, D).")

    if patch_embeddings.shape[1] != class_token.shape[1]:
        raise ValueError("Embedding dimensions (D) must match.")

    # --- Concatenate along sequence dimension ---
    return np.concatenate([class_token, patch_embeddings], axis=0)
