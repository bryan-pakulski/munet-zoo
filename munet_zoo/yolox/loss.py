# YOLOX architecture: Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (Apache-2.0). SimOTA assignment as described in the
# paper, written for MuNet as graph operations; no upstream source is included. Released under this repository's licence;
# see NOTICE.
"""YOLOX training loss: SimOTA assignment over padded target slots, IoU loss, objectness and class BCE.

Targets are fixed-size slots: ``labels`` [B, S], ``boxes`` [B, S, 4] as pixel cx, cy, w, h and ``valid`` [B, S] (1 real,
0 padding). The slot count is part of the traced program; the mask carries the box count, so any count up to S runs the
same program. The assignment is computed from detached predictions.
"""
import math
import numpy as np
from munet import nn
from .model import YOLOX

BIG = 1e5


def pad_targets(targets, slots):
    """Pad a list of {'labels': [...], 'boxes': [[cx, cy, w, h], ...]} to the slot arrays the loss takes."""
    out = {'labels': np.zeros((len(targets), slots), np.float32), 'boxes': np.zeros((len(targets), slots, 4), np.float32),
           'valid': np.zeros((len(targets), slots), np.float32)}
    for i, t in enumerate(targets):
        n = len(t['labels'])
        if n > slots: raise ValueError(f"{n} boxes do not fit {slots} target slots")
        if n: out['labels'][i, :n], out['boxes'][i, :n], out['valid'][i, :n] = t['labels'], np.reshape(t['boxes'], (n, 4)), 1
    return out


def box_iou(a, b):
    """IoU of boxes given as broadcastable (cx, cy, w, h) tuples."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    iw = ((ax + aw / 2).minimum(bx + bw / 2) - (ax - aw / 2).maximum(bx - bw / 2)).clamp(min=0)
    ih = ((ay + ah / 2).minimum(by + bh / 2) - (ay - ah / 2).maximum(by - bh / 2)).clamp(min=0)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-8)



def _image_size(n):
    side = round(math.sqrt(n * 1024 / 21))  # n = side^2 * (1/64 + 1/256 + 1/1024)
    if YOLOX.grid(side)[0].shape[0] != n: raise ValueError("pass size= for outputs that are not from a square image")
    return side


class YOLOXLoss:
    def __init__(self, num_classes, center_radius=2.5, topk_dynamic=10, iou_weight=5.0):
        self.num_classes, self.center_radius, self.topk, self.iou_weight = num_classes, center_radius, topk_dynamic, iou_weight

    def candidates(self, boxes, valid, grid, stride):
        """[B, S, N]: the anchor centre lies in the box or in the box of centre_radius * stride around its centre."""
        ax, ay = ((grid[:, 0] + .5) * stride[:, 0])[None, None], ((grid[:, 1] + .5) * stride[:, 0])[None, None]
        cx, cy, w, h = (boxes[..., i:i + 1] for i in range(4))
        inside = lambda lo_x, hi_x, lo_y, hi_y: (cx * 0 + ax > lo_x) * (cx * 0 + ax < hi_x) * (cy * 0 + ay > lo_y) * (cy * 0 + ay < hi_y)
        radius = (self.center_radius * stride[:, 0])[None, None]
        in_box = inside(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
        in_centre = inside(cx - radius, cx + radius, cy - radius, cy + radius)
        return in_box.maximum(in_centre) * valid.unsqueeze(-1)

    def assign(self, outputs, targets, cand, size):
        """Matched anchors [B, S, N] (one target per anchor) from detached outputs."""
        labels, boxes = targets['labels'], targets['boxes']
        out = outputs.detach()
        pred = YOLOX.decode(out, size).permute(0, 2, 1)  # [B, 4, N]
        iou = box_iou([pred[:, i:i + 1] for i in range(4)], [boxes[..., i:i + 1] for i in range(4)])
        p = (out[..., 5:].sigmoid() * out[..., 4:5].sigmoid()).sqrt().clamp(min=1e-6, max=1 - 1e-6)
        onehot = labels.unsqueeze(-1).eq(np.arange(self.num_classes, dtype=np.float32))
        # sum over classes of BCE(p, onehot) = sum(-log(1 - p)) + onehot . (log(1 - p) - log p)
        base = (1 - p).log().sum(-1) * -1
        cost = base.unsqueeze(1) + onehot @ ((1 - p).log() - p.log()).permute(0, 2, 1)
        cost = cost + 3 * -(iou + 1e-8).log() + BIG * (1 - cand)
        k = (iou * cand).topk(self.topk, 2)[0].sum(2, keepdim=True).floor().clamp(min=1)
        kth = -(-cost).topk(self.topk, 2)[0].gather(2, k - 1)
        match = (cost <= kth) * cand
        pick = (-(cost + 1e6 * (1 - match))).topk(1, 1)[1]  # argmin over the target slots
        return match * pick.eq(np.arange(boxes.shape[1], dtype=np.float32).reshape(1, -1, 1))

    def __call__(self, outputs, targets, size=None):
        size = size or _image_size(outputs.shape[1])
        grid, stride = YOLOX.grid(size)
        labels, boxes, valid = targets['labels'], targets['boxes'], targets['valid']
        cand = self.candidates(boxes, valid, grid, stride)
        sel = self.assign(outputs, targets, cand, size)  # [B, S, N]
        pos = sel.sum(1)  # [B, N]
        count = pos.sum().clamp(min=1)
        target_box = sel.permute(0, 2, 1) @ boxes  # [B, N, 4]
        onehot = labels.unsqueeze(-1).eq(np.arange(self.num_classes, dtype=np.float32))
        target_cls = sel.permute(0, 2, 1) @ onehot
        pred = YOLOX.decode(outputs, size)
        iou = box_iou([pred[..., i:i + 1] for i in range(4)], [target_box[..., i:i + 1] for i in range(4)])  # [B, N, 1]
        iou_loss = ((1 - iou * iou) * pos.unsqueeze(-1)).sum()
        obj = nn.functional.binary_cross_entropy_with_logits(outputs[..., 4:5], pos.unsqueeze(-1), reduction='sum')
        cls = (nn.functional.binary_cross_entropy_with_logits(outputs[..., 5:], target_cls * iou.detach(), reduction='none')
               * pos.unsqueeze(-1)).sum()
        terms = {'iou': self.iou_weight * iou_loss / count, 'obj': obj / count, 'cls': cls / count}
        return {**terms, 'total': terms['iou'] + terms['obj'] + terms['cls'], 'matched': count}
