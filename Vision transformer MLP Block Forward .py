import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation (tanh approximation used in Transformers).
    
    GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    """
    return 0.5 * x * (
        1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
        )
    )


def vit_mlp_block_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray
) -> np.ndarray:
    """
    Vision Transformer MLP block forward pass.

    Args:
        x  : (N, D)
        W1 : (D, D_hidden)
        b1 : (D_hidden,)
        W2 : (D_hidden, D)
        b2 : (D,)

    Returns:
        (N, D)
    """

    # ---------- Sanity checks ----------
    if x.ndim != 2:
        raise ValueError("x must be a 2D array (N, D).")

    if W1.shape[0] != x.shape[1]:
        raise ValueError("W1 input dimension must match x feature dimension.")

    if W2.shape[1] != x.shape[1]:
        raise ValueError("W2 output dimension must match x feature dimension.")

    # ---------- Linear 1 ----------
    hidden = x @ W1 + b1   # (N, D_hidden)

    # ---------- GELU ----------
    hidden = gelu(hidden) # (N, D_hidden)

    # ---------- Linear 2 ----------
    out = hidden @ W2 + b2  # (N, D)

    return out.astype(np.float32)


#Problem Description
#The Vision Transformer encoder uses a Multi-Layer Perceptron (MLP) block as part of each transformer layer. This MLP consists of two linear transformations with a GELU (Gaussian Error Linear Unit) activation function in between.

#The MLP block expands the dimensionality in the first layer (typically by a factor of 4), applies GELU activation, and then projects back to the original dimension.
