"""ConvNeXt (Liu et al., 2022) built from MuNet layers, including the DINOv3-distilled ConvNeXt models.

Parameter names match timm's ``convnext_*`` models, so their weights (for example ``convnext_tiny.dinov3_lvd1689m``)
load with ``load_timm``. Weights are not part of this repository; check the licence of the weights you load.
"""
import numpy as np
from munet import nn
from munet.core import Parameter
from . import register
from .convnextv2 import LayerNorm2d, read_safetensors

# variant: (blocks per stage, channels per stage)
VARIANTS = {
    'tiny': ((3, 3, 9, 3), (96, 192, 384, 768)),
    'small': ((3, 3, 27, 3), (96, 192, 384, 768)),
    'base': ((3, 3, 27, 3), (128, 256, 512, 1024)),
    'large': ((3, 3, 27, 3), (192, 384, 768, 1536)),
}


def _per_channel(parameter): return parameter.reshape(1, -1, 1, 1)


class Mlp(nn.Module):
    def __init__(self, channels):
        self.fc1 = nn.Conv2d(channels, 4 * channels, 1)
        self.fc2 = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x): return self.fc2(self.fc1(x).gelu())


class Block(nn.Module):
    def __init__(self, channels, layer_scale=1e-6):
        self.conv_dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.mlp = Mlp(channels)
        self.gamma = Parameter(np.full(channels, layer_scale, np.float32))

    def forward(self, x): return x + _per_channel(self.gamma) * self.mlp(self.norm(self.conv_dw(x)))


class Stage(nn.Module):
    def __init__(self, ch_in, ch_out, depth, downsample):
        self.downsample = (nn.Sequential(LayerNorm2d(ch_in), nn.Conv2d(ch_in, ch_out, 2, 2)) if downsample
                           else nn.Identity())
        self.blocks = nn.Sequential(*[Block(ch_out) for _ in range(depth)])

    def forward(self, x): return self.blocks(self.downsample(x))


class ConvNeXt(nn.Module):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, variant='tiny', return_idx=(1, 2, 3), freeze_at=-1, out_norm=False):
        """return_idx: stages whose output is returned (stage i has stride 4 * 2**i). freeze_at: -1 trains everything,
        n >= 0 freezes the stem and the first n stages. out_norm: a LayerNorm on every returned feature map."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown ConvNeXt variant {variant!r}; available: {sorted(VARIANTS)}")
        depths, dims = VARIANTS[variant]
        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 4, 4), LayerNorm2d(dims[0]))
        self.stages = nn.ModuleList([Stage(dims[max(i - 1, 0)], dims[i], depths[i], i > 0) for i in range(4)])
        self.return_idx = list(return_idx)
        self.out_channels = [dims[i] for i in self.return_idx]
        self.out_strides = [4 * 2 ** i for i in self.return_idx]
        self.out_norms = nn.ModuleList([LayerNorm2d(c) for c in self.out_channels]) if out_norm else None
        if freeze_at >= 0:
            self.stem.requires_grad_(False)
            for stage in self.stages.layers[:freeze_at]: stage.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx: outs.append(x)
        if self.out_norms is not None: outs = [norm(o) for norm, o in zip(self.out_norms, outs)]
        return outs

    def load_timm(self, path):
        """Load a timm ``convnext_*`` ``model.safetensors``; the classifier head and final norm are dropped."""
        state = {}
        for name, value in read_safetensors(path).items():
            if name.startswith(('head.', 'norm_pre.')): continue
            if name.endswith(('.mlp.fc1.weight', '.mlp.fc2.weight')) and value.ndim == 2:
                value = value[:, :, None, None]
            state[name] = value
        self.load_state_dict(state, strict=self.out_norms is None)
        return self


for _variant in VARIANTS:
    register(f'convnext_{_variant}')(lambda _variant=_variant, **kwargs: ConvNeXt(_variant, **kwargs))
