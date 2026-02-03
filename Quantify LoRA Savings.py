def quantify_lora_savings(out_dim: int, in_dim: int, rank: int):
    """
    Compute memory and FLOP savings for LoRA vs full fine-tuning.

    Args:
        out_dim (int): Output dimension
        in_dim (int): Input dimension
        rank (int): LoRA rank

    Returns:
        (memory_savings_pct, flops_savings_pct): tuple of floats (0–100)
    """

    # Full parameters / FLOPs
    full_params = out_dim * in_dim
    full_flops = full_params  # per forward pass for one input vector

    # LoRA parameters / FLOPs
    lora_params = rank * (out_dim + in_dim)
    lora_flops = lora_params

    # Savings percentages
    memory_savings_pct = (1 - lora_params / full_params) * 100
    flops_savings_pct = (1 - lora_flops / full_flops) * 100

    return memory_savings_pct, flops_savings_pct


#Problem Description
#One of the key advantages of LoRA is the significant reduction in parameters and computational cost compared to full fine-tuning. This problem computes the memory and FLOP savings achieved by using LoRA.

