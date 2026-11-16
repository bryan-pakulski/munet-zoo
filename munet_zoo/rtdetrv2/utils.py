# Adapted for MuNet from lyuwenyu/RT-DETR (rtdetrv2_pytorch), commit e97db5a685a874c01c7977cb3a42ab0684ea134b.
# Original work Copyright (c) 2023 lyuwenyu and RT-DETR contributors, Apache-2.0.
# See LICENSE-RT-DETR and NOTICE in this directory.
"""Helpers shared by the RT-DETRv2 encoder and decoder."""
import math

from munet import nn
from munet.core import cat

from ..backbones.common import get_activation  # noqa: F401  (re-exported)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0).clamp(max=1)  # one-sided clamps: MuNet's two-sided clamp op has no ONNX export
    return (x.clamp(min=eps) / (1 - x).clamp(min=eps)).log()


def bias_init_with_prob(prob=0.01): return -math.log((1 - prob) / prob)


def deformable_attention_core_func_v2(value, spatial_shapes, sampling_locations, attention_weights, num_points_list, method='default'):
    """Sample each level at its own number of points and sum the samples with the attention weights.

    value [b, sum(h*w), heads, c]; sampling_locations [b, q, heads, sum(points), 2] in [0, 1];
    attention_weights [b, q, heads, sum(points)]. Returns [b, q, heads*c].
    method 'default' samples bilinearly; 'discrete' reads the nearest cell.
    """
    b, _, heads, channels = value.shape
    queries, total = sampling_locations.shape[1], sum(num_points_list)
    if sum(h * w for h, w in spatial_shapes) != value.shape[1]: raise ValueError("deformable feature shapes do not cover values")
    if sampling_locations.shape[3] != total: raise ValueError("sampling locations do not match the points per level")
    values = value.split([h * w for h, w in spatial_shapes], 1)
    grids = sampling_locations.permute(0, 2, 1, 3, 4).reshape(b * heads, queries, total, 2).split(list(num_points_list), 2)
    sampled = []
    for (h, w), level, grid, points in zip(spatial_shapes, values, grids, num_points_list):
        image = level.flatten(2).permute(0, 2, 1).reshape(b * heads, channels, h, w)
        if method == 'default':
            sampled.append(nn.functional.grid_sample(image, 2 * grid - 1))
        else:
            x = (grid[..., 0] * w + 0.5).floor().clamp(min=0).clamp(max=w - 1)
            y = (grid[..., 1] * h + 0.5).floor().clamp(min=0).clamp(max=h - 1)
            index = (y * w + x).reshape(b * heads, 1, queries * points).expand(b * heads, channels, queries * points)
            sampled.append(image.reshape(b * heads, channels, h * w).gather(2, index).reshape(b * heads, channels, queries, points))
    weights = attention_weights.permute(0, 2, 1, 3).reshape(b * heads, 1, queries, total)
    return (cat(sampled, -1) * weights).sum(-1).reshape(b, heads * channels, queries).permute(0, 2, 1)
