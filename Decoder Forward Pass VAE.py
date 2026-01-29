import numpy as np

def decoder_forward(z, W_out, b_out):
    """
    Forward pass of a linear VAE decoder.

    Parameters:
    z      : (B, D_latent)
    W_out  : (D_latent, D_out)
    b_out  : (D_out,)

    Returns:
    x_recon : (B, D_out)
    """
    x_recon = z @ W_out + b_out
    return x_recon
