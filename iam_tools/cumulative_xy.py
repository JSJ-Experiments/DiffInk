"""Absolute path supervision for normalized chronological displacements.

Offsets include pen-up jumps and origin->first point. Summing displacement ERROR
in model units equals predicted-minus-target absolute XY. No endpoint correction,
target smoothing, arc-length resampling, or target information at inference.
Padding cannot contribute to the loss or its gradients. Index increments are NOT
physical velocities (the original IAM/RDP point spacing is nonuniform).
"""
import math
import torch


def cumulative_xy_loss(prediction, target, real_mask, stats):
    if prediction.shape != target.shape or prediction.shape != (*real_mask.shape, 2) or prediction.ndim != 4 or prediction.shape[2] != 8 or real_mask.dtype != torch.bool:
        raise ValueError('matched B,L,8,2 normalized offsets and Boolean real mask required')
    mask = real_mask.flatten(1)
    if not mask.any(1).all() or ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError('each line requires nonempty prefix-valid chronological points')
    std = prediction.new_tensor(stats['std'])
    if std.shape != (2,) or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('finite positive TRAIN offset axis scale required')
    error = ((prediction-target)*std).flatten(1, 2)
    error = torch.where(mask[..., None], error, 0.)
    cumulative = error.cumsum(1)
    return cumulative[mask].square().mean()


def anchor_coefficient(offset_norm, anchor_norm, fraction=.15):
    if not all(math.isfinite(v) for v in (offset_norm, anchor_norm, fraction)) or min(offset_norm, anchor_norm) <= 1e-12 or not 0 < fraction <= .25:
        raise ValueError('finite nondegenerate gradients and modest auxiliary fraction required')
    return fraction*offset_norm/anchor_norm
