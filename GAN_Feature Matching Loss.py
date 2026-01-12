import numpy as np

def feature_matching_loss(
    real_features: np.ndarray,
    fake_features: np.ndarray
) -> float:
    """
    Compute Feature Matching loss.

    Parameters
    ----------
    real_features : np.ndarray
        Discriminator features for real data.
        Shape: (batch_size_real, feature_dim)
    fake_features : np.ndarray
        Discriminator features for generated data.
        Shape: (batch_size_fake, feature_dim)

    Returns
    -------
    float
        Scalar feature matching loss
    """
    # Mean feature vectors
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    # Squared L2 norm of the difference
    loss = np.sum((mu_real - mu_fake) ** 2)

    return float(loss)


#Implement the Feature Matching loss, a technique to improve stability of GAN training. Instead of maximizing the output of the discriminator directly, the generator is trained to match the statistics (specifically the mean) of the features extracted by the discriminator on real data.

