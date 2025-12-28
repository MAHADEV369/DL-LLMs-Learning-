import numpy as np

def random_noise_sampler(
    batch_size: int,
    latent_dim: int,
    mode: str = "gaussian",
    seed: int | None = None
) -> np.ndarray:
    """
    Generate random noise vectors (latent codes) for a generator.

    Parameters
    ----------
    batch_size : int
        Number of samples to generate.
    latent_dim : int
        Dimension of the latent vector z.
    mode : str, optional
        Distribution type: 'gaussian' or 'uniform'. Default is 'gaussian'.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    noise : np.ndarray
        Array of shape (batch_size, latent_dim) with dtype np.float32.
    """
    if seed is not None:
        np.random.seed(seed)

    if mode == "gaussian":
        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(batch_size, latent_dim)
        )
    elif mode == "uniform":
        noise = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(batch_size, latent_dim)
        )
    else:
        raise ValueError("mode must be either 'gaussian' or 'uniform'")

    return noise.astype(np.float32)



#Implement a function to generate random noise vectors (latent codes) z
#z which serve as input to the Generator.

