"""xformers.ops stub – memory_efficient_attention via torch SDPA."""

import torch
import torch.nn.functional as F


class LowerTriangularMask:
    """Sentinel used by AudioCraft to request causal masking."""
    pass


def memory_efficient_attention(
    query,
    key,
    value,
    attn_bias=None,
    p: float = 0.0,
    scale: float = None,
    op=None,
):
    """Drop-in replacement using torch.nn.functional.scaled_dot_product_attention.

    AudioCraft passes tensors as (B, M, H, K) – batch, seq_len, heads, head_dim.
    PyTorch SDPA expects (B, H, M, K) – batch, heads, seq_len, head_dim.
    """
    # Transpose from (B, M, H, K) -> (B, H, M, K)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    is_causal = False
    attn_mask = None

    if isinstance(attn_bias, LowerTriangularMask):
        is_causal = True
    elif attn_bias is not None:
        attn_mask = attn_bias

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=p if query.requires_grad else 0.0,
        is_causal=is_causal,
        scale=scale,
    )

    # Transpose back from (B, H, M, K) -> (B, M, H, K)
    return out.transpose(1, 2)


def unbind(x, dim=0):
    """Drop-in for xformers.ops.unbind — delegates to torch.unbind."""
    return torch.unbind(x, dim=dim)
