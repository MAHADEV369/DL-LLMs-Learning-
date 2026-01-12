import numpy as np

def gradient_penalty(gradients: np.ndarray, lambda_gp: float = 10.0) -> float:
    """
    Compute WGAN-GP gradient penalty.

    Parameters
    ----------
    gradients : np.ndarray
        Gradient of critic output w.r.t interpolated inputs.
        Shape: (batch_size, input_dim)
    lambda_gp : float
        Gradient penalty coefficient (default 10.0)

    Returns
    -------
    float
        Scalar gradient penalty value
    """
    # L2 norm of gradients per sample
    grad_norm = np.linalg.norm(gradients, axis=1)

    # (||grad||_2 - 1)^2
    penalty_per_sample = (grad_norm - 1.0) ** 2

    # Mean over batch and scale by lambda
    penalty = lambda_gp * np.mean(penalty_per_sample)

    return float(penalty)



#Implement the gradient penalty term used in WGAN-GP (Wasserstein GAN with Gradient Penalty). This term enforces the Lipschitz constraint on the discriminator (critic) by penalizing gradients whose L2 norm deviates from 1.

