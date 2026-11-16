# The matching costs and losses follow lyuwenyu/RT-DETR (rtdetrv2_pytorch), commit
# e97db5a685a874c01c7977cb3a42ab0684ea134b, Apache-2.0. See LICENSE-RT-DETR and NOTICE in this directory.
"""RT-DETRv2 matching, varifocal/focal classification and box losses on native graphs.

Targets are padded: ``labels`` [B, S], ``boxes`` [B, S, 4] (normalized cx, cy, w, h) and ``valid`` [B, S] (1 for a real
object). ``pad_targets`` in ``denoising.py`` builds them.
"""
import numpy as np
from munet import nn
from munet.core import as_tensor, cat, operation, where


def cxcywh_to_xyxy(boxes): return cat([boxes[..., :2] - boxes[..., 2:] * 0.5, boxes[..., :2] + boxes[..., 2:] * 0.5], -1)


def box_overlap(a, b):
    """Broadcasted paired IoU/GIoU; zero-area padding has finite zero derivatives."""
    a, b = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    lo = a[..., :2].maximum(b[..., :2]); hi = a[..., 2:].minimum(b[..., 2:])
    size = (hi - lo).clamp(min=0); intersection = size[..., 0] * size[..., 1]
    sa = a[..., 2:] - a[..., :2]; sb = b[..., 2:] - b[..., :2]
    union = sa[..., 0] * sa[..., 1] + sb[..., 0] * sb[..., 1] - intersection
    iou = intersection / where(union > 0, union, 1)
    outer = b[..., 2:].maximum(a[..., 2:]) - b[..., :2].minimum(a[..., :2])
    area = outer[..., 0] * outer[..., 1]
    giou = iou - (area - union) / where(area > 0, area, 1)
    return iou, giou


def one_hot(labels, classes):
    return labels.unsqueeze(-1).eq(np.arange(classes, dtype=np.float32))


class HungarianMatcher(nn.Module):
    """Returns, per image and target slot, the index of the matched query (float32, [B, S])."""
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        self.alpha, self.gamma = alpha, gamma

    def forward(self, outputs, targets):
        logits, boxes = outputs['pred_logits'].detach(), outputs['pred_boxes'].detach()
        probabilities = logits.sigmoid()
        selected = probabilities @ one_hot(targets['labels'], logits.shape[-1]).transpose(-2, -1)
        negative = (1 - self.alpha) * (selected ** self.gamma) * (-(1 - selected + 1e-8).log())
        positive = self.alpha * ((1 - selected) ** self.gamma) * (-(selected + 1e-8).log())
        pred, truth = boxes.unsqueeze(2), targets['boxes'].unsqueeze(1)
        l1 = (pred - truth).abs().sum(-1)
        _, giou = box_overlap(pred, truth)
        cost = self.cost_class * (positive - negative) + self.cost_bbox * l1 - self.cost_giou * giou
        return operation('assignment', cost, targets['valid'])


class RTDETRCriterionv2(nn.Module):
    """classification: 'vfl' or 'focal'. boxes_weight_format: None, 'iou' or 'giou'; when set, that overlap of each matched
    pair weights its GIoU loss and is the varifocal target (None uses the IoU as target and no GIoU weight).
    share_matched_indices: the decoder's auxiliary layers reuse the last layer's matching instead of matching again."""
    def __init__(self, num_classes, weight_dict=None, alpha=0.75, gamma=2.0, classification='vfl', matcher=None,
                 boxes_weight_format=None, share_matched_indices=False):
        if classification not in ('vfl', 'focal'): raise ValueError("classification must be vfl or focal")
        if boxes_weight_format not in (None, 'iou', 'giou'): raise ValueError("boxes_weight_format must be None, iou or giou")
        self.num_classes, self.alpha, self.gamma, self.classification = num_classes, alpha, gamma, classification
        self.weight_dict = weight_dict or {'loss_' + classification: 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
        self.matcher = matcher or HungarianMatcher()
        self.boxes_weight_format, self.share_matched_indices = boxes_weight_format, share_matched_indices

    def _loss(self, outputs, targets, assignment, normalizer):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        b, q, c = logits.shape; slots = targets['labels'].shape[1]
        valid = targets['valid'] * (assignment >= 0)
        matched = boxes.gather(1, assignment.unsqueeze(-1).expand(b, slots, 4))
        iou, giou = box_overlap(matched, targets['boxes'])
        quality = (giou if self.boxes_weight_format == 'giou' else iou).detach()
        box_weight = valid if self.boxes_weight_format is None else valid * quality
        mapping = as_tensor(np.arange(q, dtype=np.float32).reshape(1, q, 1)).eq(assignment.unsqueeze(1)) * valid.unsqueeze(1)
        labels = one_hot(targets['labels'], c)
        target = mapping @ labels
        if self.classification == 'vfl':
            target_score = mapping @ (labels * quality.unsqueeze(-1))
            weight = self.alpha * (logits.sigmoid().detach() ** self.gamma) * (1 - target) + target_score
            classification = ((logits.softplus() - logits * target_score) * weight).sum() / normalizer
        else:
            probability = logits.sigmoid()
            pt = probability * target + (1 - probability) * (1 - target)
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            classification = ((logits.softplus() - logits * target) * alpha_t * ((1 - pt) ** self.gamma)).sum() / normalizer
        return {'loss_' + self.classification: classification,
                'loss_bbox': ((matched - targets['boxes']).abs() * valid.unsqueeze(-1)).sum() / normalizer,
                'loss_giou': ((1 - giou) * box_weight).sum() / normalizer}

    def forward(self, outputs, targets, num_boxes=None):
        """Returns a dict of weighted losses. num_boxes overrides the normalizer (the number of objects in the batch),
        for a batch that is split across devices."""
        normalizer = targets['valid'].sum().clamp(min=1) if num_boxes is None else as_tensor(num_boxes).clamp(min=1)
        losses = {}
        def add(out, truth, indices, norm, suffix):
            for name, value in self._loss(out, truth, indices, norm).items():
                if name in self.weight_dict: losses[name + suffix] = value * self.weight_dict[name]
        indices = self.matcher(outputs, targets)
        add(outputs, targets, indices, normalizer, '')
        for i, out in enumerate(outputs.get('aux_outputs', [])):
            add(out, targets, indices if self.share_matched_indices else self.matcher(out, targets), normalizer, f'_aux_{i}')
        if 'dn_aux_outputs' in outputs:
            meta = outputs['dn_meta']; groups = meta['dn_num_group']; slots = meta['max_gt']; b = targets['valid'].shape[0]
            if not 1 <= slots <= targets['valid'].shape[1]: raise ValueError("denoising target bucket mismatch")
            truth = {name: value[:, :slots] for name, value in targets.items()}
            truth = {name: value.repeat(1, groups, *([1] * (value.ndim - 2))) if groups > 1 else value for name, value in truth.items()}
            idx = np.concatenate([np.arange(slots, dtype=np.float32) + g * 2 * slots for g in range(groups)])
            dn_indices = as_tensor(idx.reshape(1, -1)).expand(b, -1)
            for i, out in enumerate(outputs['dn_aux_outputs']): add(out, truth, dn_indices, normalizer * groups, f'_dn_{i}')
        if 'enc_aux_outputs' in outputs:
            # a class-agnostic encoder head scores objectness only, so every object is class 0 for it
            truth = {**targets, 'labels': targets['labels'] * 0} if outputs['enc_meta']['class_agnostic'] else targets
            for i, out in enumerate(outputs['enc_aux_outputs']): add(out, truth, self.matcher(out, truth), normalizer, f'_enc_{i}')
        return losses

    @staticmethod
    def total(losses): return sum(losses.values())
