import numpy as np

def vit_encoder_forward(x, layers_params):
    """
    Vision Transformer Encoder Forward (Pure NumPy, Pre-LN)

    x: (N, D) float32
    layers_params: list of encoder-layer parameter dicts
    returns: (N, D) float32
    """

    def layer_norm(x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        ))

    def multi_head_self_attention(x, p):
        W_q, W_k, W_v, W_o = p['W_q'], p['W_k'], p['W_v'], p['W_o']
        h = p['num_heads']
        N, D = x.shape
        d = D // h

        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v

        Q = Q.reshape(N, h, d).transpose(1, 0, 2)
        K = K.reshape(N, h, d).transpose(1, 0, 2)
        V = V.reshape(N, h, d).transpose(1, 0, 2)

        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d)
        scores -= scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(axis=-1, keepdims=True)

        out = attn @ V
        out = out.transpose(1, 0, 2).reshape(N, D)
        return out @ W_o

    def mlp(x, p):
        return gelu(x @ p['W1'] + p['b1']) @ p['W2'] + p['b2']

    # ----- Encoder stack -----
    for lp in layers_params:
        # Attention block (Pre-LN)
        x = x + multi_head_self_attention(
            layer_norm(x, lp['ln1_params']['gamma'], lp['ln1_params']['beta']),
            lp['attn_params']
        )

        # MLP block (Pre-LN)
        x = x + mlp(
            layer_norm(x, lp['ln2_params']['gamma'], lp['ln2_params']['beta']),
            lp['mlp_params']
        )

    return x.astype(np.float32)


#ViT Encoder Forward
Problem Description
The Vision Transformer encoder consists of a stack of identical encoder layers. Each layer processes the sequence of embeddings, and the output of one layer becomes the input to the next layer.

This sequential stacking of layers allows the model to build increasingly complex representations of the input image.


