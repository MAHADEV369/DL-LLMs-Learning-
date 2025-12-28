import numpy as np

def generator_forward(z, weights, biases):
    """
    Forward pass of a linear Generator network.

    Parameters
    ----------
    z : np.ndarray
        Latent input of shape (batch_size, input_dim)
    weights : list of np.ndarray
        Weight matrices [W0, W1, ..., W_{N-1}]
        Each Wi has shape (in_dim, out_dim)
    biases : list of np.ndarray
        Bias vectors [b0, b1, ..., b_{N-1}]
        Each bi has shape (out_dim,)

    Returns
    -------
    output : np.ndarray
        Final generator output of shape (batch_size, final_out_dim)
    activations : list of np.ndarray
        List of activations [h0, h1, ..., hN]
        h0 = z, hN = output
    """
    activations = [z]
    h = z

    num_layers = len(weights)

    for i in range(num_layers):
        h = h @ weights[i] + biases[i]

        if i < num_layers - 1:
            # ReLU for hidden layers
            h = np.maximum(0, h)
        else:
            # Tanh for final layer
            h = np.tanh(h)

        activations.append(h)

    output = h
    return output, activations


#Implement the forward pass of a Generator network using only linear (fully-connected) layers and activation functions.

