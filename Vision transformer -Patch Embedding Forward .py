import numpy as np

def patch_embedding_forward(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Extracts non-overlapping patches from an image and flattens them.

    Args:
        image (np.ndarray): Input image of shape (H, W, C), dtype float32.
        patch_size (int): Size P of each square patch.

    Returns:
        np.ndarray: Patch embeddings of shape (num_patches, P*P*C), dtype float32.
    """
    H, W, C = image.shape
    P = patch_size

    # Validate divisibility
    assert H % P == 0 and W % P == 0, "H and W must be divisible by patch_size"

    # Number of patches along height and width
    num_patches_h = H // P
    num_patches_w = W // P

    # Step 1: reshape to expose patches
    # (H, W, C) → (num_patches_h, P, num_patches_w, P, C)
    patches = image.reshape(
        num_patches_h, P,
        num_patches_w, P,
        C
    )

    # Step 2: reorder axes to row-major patch order
    # → (num_patches_h, num_patches_w, P, P, C)
    patches = patches.transpose(0, 2, 1, 3, 4)

    # Step 3: flatten each patch
    # → (num_patches, P*P*C)
    patches = patches.reshape(
        num_patches_h * num_patches_w,
        P * P * C
    )

    return patches.astype(np.float32)


#In the Vision Transformer (ViT), an image is split into fixed-size patches, which are then linearly embedded. This is analogous to how words are tokenized in NLP transformers. Each patch is treated as a "token" in the sequence.

#The patch embedding layer extracts non-overlapping patches from a 2D image and flattens each patch into a vector. This creates a sequence of patch embeddings that can be processed by the transformer encoder.