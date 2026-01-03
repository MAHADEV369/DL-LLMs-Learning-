import numpy as np

def discriminator_backward(d_output, activations, weights, biases):
    """
    Backward pass for Discriminator (Linear layers only)

    Inputs
    ------
    d_output : (batch_size, 1)
        Gradient of loss w.r.t discriminator output
    activations : list of np.ndarray
        [h_0, h_1, ..., h_N]
    weights : list of np.ndarray
        Weight matrices
    biases : list of np.ndarray
        Bias vectors

    Returns
    -------
    grads_w : list of np.ndarray
        Gradients for weights
    grads_b : list of np.ndarray
        Gradients for biases
    """

    N = len(weights)  # number of layers

    grads_w = [None] * N
    grads_b = [None] * N

    # ---------- Last Layer (Sigmoid) ----------
    h_N = activations[-1]  # (batch_size, 1)
    delta = d_output * h_N * (1 - h_N)  # δ_N

    # ---------- Backpropagation ----------
    for l in reversed(range(N)):
        h_l = activations[l]  # input to layer l

        # Gradients
        grads_w[l] = h_l.T @ delta
        grads_b[l] = np.sum(delta, axis=0)

        if l > 0:
            # Propagate delta backward
            delta = delta @ weights[l].T

            # LeakyReLU derivative (applied on h_l)
            leaky_grad = np.where(h_l >= 0, 1.0, 0.2)
            delta = delta * leaky_grad

    return grads_w, grads_b






#https://papercode.in/papers/generative_adversarial_networks/problems/e07_discriminator_backward_linear
