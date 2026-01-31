import numpy as np

def build_input_embeddings(
    token_ids,
    segment_ids,
    position_embeddings,
    token_embeddings,
    segment_embeddings
):
    """
    Build BERT-style input embeddings by summing:
    token + segment + position embeddings.

    Returns:
        np.ndarray of shape (seq_len, embed_dim), dtype float32
    """

    # Lookup token embeddings
    token_embeds = token_embeddings[token_ids]          # (seq_len, embed_dim)

    # Lookup segment embeddings
    segment_embeds = segment_embeddings[segment_ids]    # (seq_len, embed_dim)

    # Sum all three components
    input_embeddings = token_embeds + segment_embeds + position_embeddings

    return input_embeddings.astype(np.float32)



#Build Input Embeddings
Problem Description
In BERT, the input representation is constructed by summing three types of embeddings:

Token embeddings: Embeddings for each token in the sequence
Segment embeddings: Embeddings indicating which segment (sentence A or B) each token belongs to
Position embeddings: Embeddings for the position of each token in the sequence
The final input embedding for each token position is the element-wise sum of these three embeddings.
