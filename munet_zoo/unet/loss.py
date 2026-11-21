"""Segmentation loss: cross entropy plus soft Dice over the classes."""
import numpy as np
import munet as mu
from munet.nn import functional as F


def segmentation_loss(logits, mask, dice_weight=1.0, eps=1.0):
    """logits (N, C, H, W), mask (N, H, W) of class indices. Returns one scalar."""
    ce = F.cross_entropy(logits, mask)
    probs = logits.softmax(1)
    classes = mu.as_tensor(np.arange(logits.shape[1], dtype=np.float32).reshape(1, -1, 1, 1))
    onehot = mask.unsqueeze(1).eq(classes)
    inter = (probs * onehot).sum((0, 2, 3))
    total = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = 1 - ((2 * inter + eps) / (total + eps)).mean()
    return ce + dice_weight * dice
