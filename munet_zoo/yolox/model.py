# YOLOX architecture: Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (Apache-2.0). Written for MuNet from the paper's
# design; no upstream source is included. Released under this repository's licence; see NOTICE.
"""YOLOX model: CSPDarknet, PAFPN and a decoupled head over strides 8, 16 and 32.

Images are NCHW RGB float32 in [0, 1] with height and width divisible by 32. The output is [B, N, 5 + classes] over every
anchor of the three strides, coarsest last. Columns: raw x, y, w, h, the objectness logit, then the class logits.
``grid`` gives the anchor centres and strides; ``decode`` turns raw boxes into pixel cx, cy, w, h.
"""
import math
import numpy as np
from munet import nn
from munet.core import cat

from ..backbones.common import ConvNormLayer

# size: (depth multiplier, width multiplier)
SIZES = {'nano': (0.33, 0.25), 'tiny': (0.33, 0.375), 's': (0.33, 0.5), 'm': (0.67, 0.75), 'l': (1.0, 1.0)}
STRIDES = (8, 16, 32)


def yolox_config(size='s', depth=None, width=None):
    """Depth and width multipliers of a named size; depth and width override them."""
    if size not in SIZES: raise ValueError(f"unknown YOLOX size {size!r}; choose from {sorted(SIZES)}")
    d, w = SIZES[size]
    return {'depth': d if depth is None else depth, 'width': w if width is None else width}


def conv(cin, cout, k=1, s=1): return ConvNormLayer(cin, cout, k, s, act='silu')


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        self.conv1, self.conv2, self.shortcut = conv(c, c, 1), conv(c, c, 3), shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class CSPLayer(nn.Module):
    """Two 1x1 branches; one runs ``n`` bottlenecks. They are joined and projected."""
    def __init__(self, cin, cout, n=1, shortcut=True):
        hidden = cout // 2
        self.conv1, self.conv2, self.conv3 = conv(cin, hidden), conv(cin, hidden), conv(2 * hidden, cout)
        self.blocks = nn.Sequential(*[Bottleneck(hidden, shortcut) for _ in range(n)])

    def forward(self, x): return self.conv3(cat([self.blocks(self.conv1(x)), self.conv2(x)], 1))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, kernels=(5, 9, 13)):
        self.conv1, self.conv2, self.kernels = conv(cin, cin // 2), conv(cin * 2, cout), kernels

    def forward(self, x):
        x = self.conv1(x)
        pools = [nn.functional.max_pool2d(x, k, 1, k // 2) for k in self.kernels]
        return self.conv2(cat([x, *pools], 1))


class Focus(nn.Module):
    """Space to depth (four interleaved samplings), then a convolution."""
    def __init__(self, cin, cout, k=3): self.conv = conv(4 * cin, cout, k)

    def forward(self, x):
        return self.conv(cat([x[:, :, ::2, ::2], x[:, :, 1::2, ::2], x[:, :, ::2, 1::2], x[:, :, 1::2, 1::2]], 1))


def stage(cin, cout, n, spp=False):
    layers = [conv(cin, cout, 3, 2)] + ([SPPBottleneck(cout, cout)] if spp else [])
    return nn.Sequential(*layers, CSPLayer(cout, cout, n, shortcut=not spp))


class CSPDarknet(nn.Module):
    def __init__(self, depth, width):
        c, n = int(width * 64), max(round(depth * 3), 1)
        self.stem = Focus(3, c)
        self.dark2, self.dark3 = stage(c, 2 * c, n), stage(2 * c, 4 * c, 3 * n)
        self.dark4, self.dark5 = stage(4 * c, 8 * c, 3 * n), stage(8 * c, 16 * c, n, spp=True)
        self.out_channels = [4 * c, 8 * c, 16 * c]

    def forward(self, x):
        c3 = self.dark3(self.dark2(self.stem(x)))
        c4 = self.dark4(c3)
        return [c3, c4, self.dark5(c4)]


def up(x): return nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class PAFPN(nn.Module):
    """Top-down fusion from the coarsest level, then bottom-up fusion back up."""
    def __init__(self, depth, width):
        c, n = int(width * 64), max(round(depth * 3), 1)
        self.lateral0, self.lateral1 = conv(16 * c, 8 * c), conv(8 * c, 4 * c)
        self.td4, self.td3 = CSPLayer(16 * c, 8 * c, n, False), CSPLayer(8 * c, 4 * c, n, False)
        self.down3, self.down4 = conv(4 * c, 4 * c, 3, 2), conv(8 * c, 8 * c, 3, 2)
        self.bu4, self.bu5 = CSPLayer(8 * c, 8 * c, n, False), CSPLayer(16 * c, 16 * c, n, False)

    def forward(self, feats):
        c3, c4, c5 = feats
        p5 = self.lateral0(c5)
        p4 = self.lateral1(self.td4(cat([up(p5), c4], 1)))
        out3 = self.td3(cat([up(p4), c3], 1))
        out4 = self.bu4(cat([self.down3(out3), p4], 1))
        out5 = self.bu5(cat([self.down4(out4), p5], 1))
        return [out3, out4, out5]


class HeadLevel(nn.Module):
    """Shared 1x1 stem, then a class branch and a box/objectness branch of two 3x3 convolutions each."""
    def __init__(self, cin, hidden, num_classes):
        self.stem = conv(cin, hidden)
        self.cls_convs = nn.Sequential(conv(hidden, hidden, 3), conv(hidden, hidden, 3))
        self.reg_convs = nn.Sequential(conv(hidden, hidden, 3), conv(hidden, hidden, 3))
        self.cls_pred, self.reg_pred, self.obj_pred = (nn.Conv2d(hidden, k, 1) for k in (num_classes, 4, 1))
        prior = -math.log(99.0)  # objectness and class probability start at 0.01
        for p in (self.cls_pred.bias, self.obj_pred.bias): p.assign(np.full(p.shape, prior, np.float32))

    def forward(self, x):
        x = self.stem(x)
        r, c = self.reg_convs(x), self.cls_convs(x)
        y = cat([self.reg_pred(r), self.obj_pred(r), self.cls_pred(c)], 1)
        return y.flatten(2).permute(0, 2, 1)


class YOLOX(nn.Module):
    """``size`` is nano, tiny, s, m or l; ``depth`` and ``width`` override its multipliers."""
    def __init__(self, num_classes=80, size='s', depth=None, width=None, *, seed=0):
        if num_classes <= 0: raise ValueError("num_classes must be positive")
        mult = yolox_config(size, depth, width)
        self.config = dict(num_classes=num_classes, size=size, **mult, seed=seed)
        self.num_classes = num_classes
        token = nn._initialization_rng.set(np.random.default_rng(seed))
        try:
            self.backbone, self.neck = CSPDarknet(**mult), PAFPN(**mult)
            hidden = int(256 * mult['width'])
            self.head = nn.ModuleList([HeadLevel(c, hidden, num_classes) for c in self.backbone.out_channels])
        finally: nn._initialization_rng.reset(token)
        self.max_stride = max(STRIDES)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or any(s % self.max_stride for s in images.shape[-2:]):
            raise ValueError(f"YOLOX expects NCHW RGB images with dimensions divisible by {self.max_stride}")
        feats = self.neck(self.backbone(images))
        return cat([level(f) for level, f in zip(self.head, feats)], 1)

    @staticmethod
    def grid(size, strides=STRIDES):
        """Anchor centres [N, 2] in cells (x, y) and strides [N, 1] for an image of ``size`` (an int or height, width)."""
        h, w = (size, size) if isinstance(size, int) else size
        grids, steps = [], []
        for s in strides:
            ys, xs = np.meshgrid(np.arange(h // s), np.arange(w // s), indexing='ij')
            grids.append(np.stack([xs, ys], -1).reshape(-1, 2)); steps.append(np.full((len(grids[-1]), 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(steps).astype(np.float32)

    @staticmethod
    def decode(outputs, size):
        """Boxes [B, N, 4] as pixel cx, cy, w, h: xy = (raw xy + cell) * stride, wh = exp(raw wh) * stride."""
        grid, stride = YOLOX.grid(size)
        return cat([(outputs[..., :2] + grid) * stride, outputs[..., 2:4].exp() * stride], -1)
