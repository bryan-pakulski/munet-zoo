"""RT-DETRv2: a swappable backbone, the hybrid encoder and the iterative deformable decoder."""
import numpy as np
from munet import nn

from ..backbones import build_backbone
from .decoder import RTDETRTransformerv2
from .encoder import HybridEncoder


class RTDETRv2(nn.Module):
    """backbone is a name from ``munet_zoo.backbones.available()``; backbone_args are passed to its factory.

    Images are NCHW RGB float32 in [0, 1]. The model applies the backbone's own mean/std, if it has them.
    In eval mode the output is ``pred_logits`` [B, Q, classes] and ``pred_boxes`` [B, Q, 4] (normalized cx, cy, w, h).
    """
    def __init__(self, num_classes=80, *, backbone='convnextv2_atto', backbone_args=None, hidden_dim=256, nhead=8,
                 num_queries=300, num_decoder_layers=6, num_points=4, dim_feedforward=1024, num_denoising=100,
                 expansion=1.0, depth_mult=1.0, cross_attn_method='default', query_select_method='default',
                 learn_query_content=False, eval_spatial_size=None, seed=0):
        if num_classes <= 0 or num_queries <= 0 or num_decoder_layers <= 0: raise ValueError("invalid detector dimensions")
        if hidden_dim % 4 or hidden_dim % nhead: raise ValueError("hidden dimension must be divisible by four and the head count")
        self.config = dict(num_classes=num_classes, backbone=backbone, backbone_args=dict(backbone_args or {}), hidden_dim=hidden_dim,
                           nhead=nhead, num_queries=num_queries, num_decoder_layers=num_decoder_layers,
                           num_points=list(num_points) if isinstance(num_points, (list, tuple)) else num_points,
                           dim_feedforward=dim_feedforward, num_denoising=num_denoising, expansion=expansion, depth_mult=depth_mult,
                           cross_attn_method=cross_attn_method, query_select_method=query_select_method,
                           learn_query_content=learn_query_content,
                           eval_spatial_size=None if eval_spatial_size is None else list(eval_spatial_size), seed=seed)
        self.num_classes, self.num_queries = num_classes, num_queries
        token = nn._initialization_rng.set(np.random.default_rng(seed))
        try:
            self.backbone = build_backbone(backbone, **self.config['backbone_args'])
            strides, levels = list(self.backbone.out_strides), len(self.backbone.out_strides)
            self.encoder = HybridEncoder(in_channels=list(self.backbone.out_channels), feat_strides=strides, hidden_dim=hidden_dim,
                                         nhead=nhead, dim_feedforward=dim_feedforward, use_encoder_idx=[levels - 1],
                                         expansion=expansion, depth_mult=depth_mult, eval_spatial_size=eval_spatial_size)
            self.decoder = RTDETRTransformerv2(num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries,
                                               feat_channels=[hidden_dim] * levels, feat_strides=strides, num_levels=levels,
                                               num_points=num_points, nhead=nhead, num_layers=num_decoder_layers,
                                               dim_feedforward=dim_feedforward, num_denoising=num_denoising,
                                               learn_query_content=learn_query_content, eval_spatial_size=eval_spatial_size,
                                               cross_attn_method=cross_attn_method, query_select_method=query_select_method)
        finally: nn._initialization_rng.reset(token)
        self.max_stride = max(strides)
        mean, std = getattr(self.backbone, 'mean', None), getattr(self.backbone, 'std', None)
        self.mean = None if mean is None else np.asarray(mean, np.float32).reshape(1, 3, 1, 1)
        self.std = None if std is None else np.asarray(std, np.float32).reshape(1, 3, 1, 1)

    def forward(self, images, denoising=None):
        if images.ndim != 4 or images.shape[1] != 3 or any(s % self.max_stride for s in images.shape[-2:]):
            raise ValueError(f"RT-DETRv2 expects NCHW RGB images with dimensions divisible by {self.max_stride}")
        size = self.config['eval_spatial_size']
        if size is not None and not self.training and tuple(images.shape[-2:]) != tuple(size):
            raise ValueError("image size does not match eval_spatial_size")
        if self.mean is not None: images = (images - self.mean) / self.std
        features = self.encoder(self.backbone(images))
        if sum(x.shape[2] * x.shape[3] for x in features) < self.num_queries:
            raise ValueError("feature maps have fewer positions than queries")
        return self.decoder(features, denoising=denoising)


from .loss import RTDETRCriterionv2, HungarianMatcher  # noqa: E402
from .denoising import pad_targets, prepare_denoising  # noqa: E402
from .training import DetectorTrainer, rtdetrv2_adamw, load_torch_checkpoint  # noqa: E402
from .stages import pool_stages  # noqa: E402
from .data import PostProcessor, Predictor, CocoDetection, preprocess, coco_results, evaluate_coco  # noqa: E402
