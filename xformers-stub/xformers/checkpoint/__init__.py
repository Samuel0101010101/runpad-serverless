"""xformers.checkpoint stub – falls back to torch.utils.checkpoint."""

import torch.utils.checkpoint as _torch_ckpt


def checkpoint(fn, *args, **kwargs):
    """Drop-in replacement that delegates to torch.utils.checkpoint."""
    # torch.utils.checkpoint.checkpoint expects use_reentrant kwarg in newer
    # PyTorch; default to False for safety.
    kwargs.setdefault("use_reentrant", False)
    return _torch_ckpt.checkpoint(fn, *args, **kwargs)


__all__ = ["checkpoint"]
