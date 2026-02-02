import numpy as np

def reparameterization_trick(mu, logvar, epsilon):
    """
    Reparameterization trick for VAEs.

    Parameters:
    mu : np.ndarray, shape (d,), dtype float32
    logvar : np.ndarray, shape (d,), dtype float32
    epsilon : np.ndarray, shape (d,), dtype float32

    Returns:
    z : np.ndarray, shape (d,), dtype float32
    """
    std = np.exp(0.5 * logvar)
    z = mu + std * epsilon
    return z.astype(np.float32)


#Problem Description
#In Variational Autoencoders (VAEs), we need to sample from a latent distribution parameterized by mean (mu) and log-variance (logvar). However, direct sampling is not differentiable. The reparameterization trick allows us to sample from the distribution in a way that maintains differentiability by separating the stochastic part (epsilon) from the deterministic parameters.

#In the World Models paper, this trick is used in the VAE encoder to generate latent representations that can be backpropagated through.
