import numpy as np

def bce_loss(y_pred, y_true):
    """
    Compute Binary Cross Entropy loss.
    
    Args:
        y_pred: NumPy array of predicted probabilities, shape (batch_size, 1)
        y_true: NumPy array of true labels, shape (batch_size, 1)
    
    Returns:
        loss: Scalar float representing the mean BCE loss over the batch
    """
    # Small epsilon for numerical stability
    epsilon = 1e-8
    
    # Compute BCE loss
    # L = -1/N * sum[y * log(y_pred + eps) + (1 - y) * log(1 - y_pred + eps)]
    term1 = y_true * np.log(y_pred + epsilon)
    term2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
    
    loss = -np.mean(term1 + term2)
    
    return float(loss)

#zBinary Cross Entropy Loss (Forward)
#Implement the Binary Cross Entropy (BCE) loss function, widely used in GANs for both the Generator and Discriminator loss terms.

