"""D4 transformations for 9x9 game tensors and policy planes."""

from __future__ import annotations

import torch


def transform_spatial(tensor: torch.Tensor, transform: int) -> torch.Tensor:
    """Rotate 0..3 times counter-clockwise, optionally after mirroring columns."""
    if transform not in range(8):
        raise ValueError("D4 transform must be in 0..7")
    mirrored, rotations = divmod(transform, 4)
    if mirrored:
        tensor = torch.flip(tensor, dims=(-1,))
    return torch.rot90(tensor, rotations, dims=(-2, -1))


def inverse_spatial(tensor: torch.Tensor, transform: int) -> torch.Tensor:
    """Undo a D4 transform of the last two dimensions."""
    if transform not in range(8):
        raise ValueError("D4 transform must be in 0..7")
    mirrored, rotations = divmod(transform, 4)
    tensor = torch.rot90(tensor, -rotations, dims=(-2, -1))
    return torch.flip(tensor, dims=(-1,)) if mirrored else tensor


def inverse_policy(logits: torch.Tensor, transform: int) -> torch.Tensor:
    if logits.shape[-1] != 81:
        raise ValueError("policy must have 81 actions")
    return inverse_spatial(logits.reshape(*logits.shape[:-1], 9, 9), transform).reshape(
        *logits.shape[:-1], 81
    )
