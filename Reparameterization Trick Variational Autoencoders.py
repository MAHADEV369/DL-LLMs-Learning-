import numpy as np

def reparameterize(mu, logvar, seed=None):
    """
    Reparameterization trick for VAEs.

    Parameters:
    mu      : (B, D_latent)
    logvar  : (B, D_latent)
    seed    : int or None

    Returns:
    z       : (B, D_latent)
    """
    if seed is not None:
        np.random.seed(seed)

    std = np.exp(0.5 * logvar)
    eps = np.random.randn(*mu.shape)
    z = mu + std * eps
    return z
