import numpy as np

def vit_classification_head(encoder_output, W_cls, b_cls):
    """
    ViT Classification Head (Pure NumPy)

    encoder_output: (N+1, D) float32
    W_cls: (D, K) float32
    b_cls: (K,) float32

    returns: (K,) float32 logits
    """
    # Extract class token (first token)
    z_class = encoder_output[0]          # (D,)

    # Linear classification layer
    logits = z_class @ W_cls + b_cls     # (K,)

    return logits.astype(np.float32)


##Problem Description
#After the encoder processes the sequence of embeddings, the Vision Transformer uses the representation of the class token (at position 0) for image classification. A simple linear layer projects this representation to the number of output classes.

#This is analogous to how BERT uses the [CLS] token for sentence classification tasks

