import numpy as np

EPS = 1e-5


# ---------------- LayerNorm ----------------
def layer_norm(x, gamma, beta, eps=EPS):
    """
    x: (N, D)
    gamma, beta: (D,)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


# ---------------- GELU ----------------
def gelu(x):
    return 0.5 * x * (
        1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        )
    )


# ---------------- MLP Block ----------------
def vit_mlp_block_forward(x, W1, b1, W2, b2):
    hidden = x @ W1 + b1
    hidden = gelu(hidden)
    out = hidden @ W2 + b2
    return out


# ---------------- Multi-Head Self-Attention ----------------
def multi_head_self_attention(x, W_q, W_k, W_v, W_o, num_heads):
    """
    x: (N, D)
    """
    N, D = x.shape
    assert D % num_heads == 0
    head_dim = D // num_heads

    # Linear projections
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    # Reshape to heads
    Q = Q.reshape(N, num_heads, head_dim)
    K = K.reshape(N, num_heads, head_dim)
    V = V.reshape(N, num_heads, head_dim)

    # Transpose for attention: (heads, N, head_dim)
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Scaled dot-product attention
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(head_dim)
    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = attn / np.sum(attn, axis=-1, keepdims=True)

    out = attn @ V  # (heads, N, head_dim)

    # Concatenate heads
    out = out.transpose(1, 0, 2).reshape(N, D)

    # Output projection
    return out @ W_o


# ---------------- ViT Encoder Layer ----------------
def vit_encoder_layer_forward(x, attn_params, mlp_params, ln1_params, ln2_params):
    """
    x: (N, D)
    """

    # ---- Pre-LN + Attention ----
    x_norm1 = layer_norm(x, ln1_params['gamma'], ln1_params['beta'])
    attn_out = multi_head_self_attention(
        x_norm1,
        attn_params['W_q'],
        attn_params['W_k'],
        attn_params['W_v'],
        attn_params['W_o'],
        attn_params['num_heads']
    )
    x_prime = x + attn_out

    # ---- Pre-LN + MLP ----
    x_norm2 = layer_norm(x_prime, ln2_params['gamma'], ln2_params['beta'])
    mlp_out = vit_mlp_block_forward(
        x_norm2,
        mlp_params['W1'],
        mlp_params['b1'],
        mlp_params['W2'],
        mlp_params['b2']
    )

    y = x_prime + mlp_out
    return y.astype(np.float32)
