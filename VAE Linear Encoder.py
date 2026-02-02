import numpy as np

def vae_encode_linear(x, W_mu, b_mu, W_logvar, b_logvar):
    """
    Linear VAE encoder projection.

    Parameters:
    x : np.ndarray, shape (d_in,), dtype float32
    W_mu : np.ndarray, shape (d_latent, d_in), dtype float32
    b_mu : np.ndarray, shape (d_latent,), dtype float32
    W_logvar : np.ndarray, shape (d_latent, d_in), dtype float32
    b_logvar : np.ndarray, shape (d_latent,), dtype float32

    Returns:
    mu : np.ndarray, shape (d_latent,), dtype float32
    logvar : np.ndarray, shape (d_latent,), dtype float32
    """
    mu = W_mu @ x + b_mu
    logvar = W_logvar @ x + b_logvar
    return mu.astype(np.float32), logvar.astype(np.float32)


#Problem Description
#In the World Models paper, the Vision (V) model is a Variational Autoencoder (VAE) that compresses high-dimensional observations into a low-dimensional latent representation. The encoder portion of the VAE maps an input observation to parameters of a latent distribution: the mean (mu) and log-variance (logvar).

#This problem focuses on implementing the linear projection step of a VAE encoder that produces these distribution parameters from a flattened input vector.

