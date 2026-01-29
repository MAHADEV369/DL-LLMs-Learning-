import numpy as np

def encoder_forward(x, W_mu, b_mu, W_logvar, b_logvar):
    """
    Forward pass of a VAE encoder.

    Parameters:
    x          : (B, D_in)
    W_mu       : (D_in, D_latent)
    b_mu       : (D_latent,)
    W_logvar   : (D_in, D_latent)
    b_logvar   : (D_latent,)

    Returns:
    mu         : (B, D_latent)
    logvar     : (B, D_latent)
    """
    mu = x @ W_mu + b_mu
    logvar = x @ W_logvar + b_logvar
    return mu, logvar
