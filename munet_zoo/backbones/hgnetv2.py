"""PP-HGNetV2 (the RT-DETR / D-FINE backbone) built from MuNet layers.

Parameter names match timm's ``hgnetv2_b*`` models, so their weights (for example ``hgnetv2_b4.ssld_stage2_ft_in1k``)
load with ``load_timm``. Weights are not part of this repository; check the licence of the weights you load.
"""
import numpy as np
from munet import nn
from munet.core import Parameter, cat
from munet.nn import functional as F
from . import register
from .convnextv2 import read_safetensors

# stage row: in, mid, out, blocks, downsample, light block, kernel, layers per block
VARIANTS = {
    'b0': ((16, 16), [[16, 16, 64, 1, False, False, 3, 3], [64, 32, 256, 1, True, False, 3, 3],
                      [256, 64, 512, 2, True, True, 5, 3], [512, 128, 1024, 1, True, True, 5, 3]]),
    'b1': ((24, 32), [[32, 32, 64, 1, False, False, 3, 3], [64, 48, 256, 1, True, False, 3, 3],
                      [256, 96, 512, 2, True, True, 5, 3], [512, 192, 1024, 1, True, True, 5, 3]]),
    'b2': ((24, 32), [[32, 32, 96, 1, False, False, 3, 4], [96, 64, 384, 1, True, False, 3, 4],
                      [384, 128, 768, 3, True, True, 5, 4], [768, 256, 1536, 1, True, True, 5, 4]]),
    'b3': ((24, 32), [[32, 32, 128, 1, False, False, 3, 5], [128, 64, 512, 1, True, False, 3, 5],
                      [512, 128, 1024, 3, True, True, 5, 5], [1024, 256, 2048, 1, True, True, 5, 5]]),
    'b4': ((32, 48), [[48, 48, 128, 1, False, False, 3, 6], [128, 96, 512, 1, True, False, 3, 6],
                      [512, 192, 1024, 3, True, True, 5, 6], [1024, 384, 2048, 1, True, True, 5, 6]]),
    'b5': ((32, 64), [[64, 64, 128, 1, False, False, 3, 6], [128, 128, 512, 2, True, False, 3, 6],
                      [512, 256, 1024, 5, True, True, 5, 6], [1024, 512, 2048, 2, True, True, 5, 6]]),
    'b6': ((48, 96), [[96, 96, 192, 2, False, False, 3, 6], [192, 192, 512, 3, True, False, 3, 6],
                      [512, 384, 1024, 6, True, True, 5, 6], [1024, 768, 2048, 3, True, True, 5, 6]]),
}
LEARNABLE_AFFINE = {'b0', 'b1', 'b2', 'b3'}  # timm turns use_lab on for these sizes only


class LearnableAffineBlock(nn.Module):
    def __init__(self):
        self.scale = Parameter(np.ones(1, np.float32))
        self.bias = Parameter(np.zeros(1, np.float32))

    def forward(self, x): return x * self.scale + self.bias


class ConvBNAct(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride=1, groups=1, act=True, lab=False):
        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride, padding=(kernel - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU() if act else nn.Identity()
        self.lab = LearnableAffineBlock() if act and lab else nn.Identity()

    def forward(self, x): return self.lab(self.act(self.bn(self.conv(x))))


class LightConvBNAct(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, lab=False):
        self.conv1 = ConvBNAct(ch_in, ch_out, 1, act=False, lab=lab)
        self.conv2 = ConvBNAct(ch_out, ch_out, kernel, groups=ch_out, lab=lab)

    def forward(self, x): return self.conv2(self.conv1(x))


class Stem(nn.Module):
    def __init__(self, mid, out, lab):
        self.stem1 = ConvBNAct(3, mid, 3, 2, lab=lab)
        self.stem2a = ConvBNAct(mid, mid // 2, 2, lab=lab)
        self.stem2b = ConvBNAct(mid // 2, mid, 2, lab=lab)
        self.stem3 = ConvBNAct(mid * 2, mid, 3, 2, lab=lab)
        self.stem4 = ConvBNAct(mid, out, 1, lab=lab)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, x):
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        x2 = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        return self.stem4(self.stem3(cat([self.pool(x), x2], dim=1)))


class Block(nn.Module):
    def __init__(self, ch_in, mid, out, layers, kernel, residual, light, lab):
        def layer(i, o): return LightConvBNAct(i, o, kernel, lab) if light else ConvBNAct(i, o, kernel, lab=lab)
        self.layers = nn.ModuleList([layer(ch_in if i == 0 else mid, mid) for i in range(layers)])
        self.aggregation = nn.Sequential(ConvBNAct(ch_in + layers * mid, out // 2, 1, lab=lab),
                                         ConvBNAct(out // 2, out, 1, lab=lab))
        self.residual = residual

    def forward(self, x):
        outs = [x]
        for layer in self.layers: outs.append(layer(outs[-1]))
        y = self.aggregation(cat(outs, dim=1))
        return y + x if self.residual else y


class Stage(nn.Module):
    def __init__(self, ch_in, mid, out, blocks, downsample, light, kernel, layers, lab):
        self.downsample = (ConvBNAct(ch_in, ch_in, 3, 2, groups=ch_in, act=False, lab=lab) if downsample
                           else nn.Identity())
        self.blocks = nn.Sequential(*[Block(ch_in if i == 0 else out, mid, out, layers, kernel, i > 0, light, lab)
                                      for i in range(blocks)])

    def forward(self, x): return self.blocks(self.downsample(x))


class HGNetV2(nn.Module):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, variant='b4', return_idx=(1, 2, 3), freeze_at=-1):
        """return_idx: stages whose output is returned (stage i has stride 4 * 2**i). freeze_at: -1 trains everything,
        n >= 0 freezes the stem and the first n stages."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown HGNetV2 variant {variant!r}; available: {sorted(VARIANTS)}")
        (mid, out), rows = VARIANTS[variant]
        lab = variant in LEARNABLE_AFFINE
        self.stem = Stem(mid, out, lab)
        self.stages = nn.ModuleList([Stage(*row, lab=lab) for row in rows])
        self.return_idx = list(return_idx)
        self.out_channels = [rows[i][2] for i in self.return_idx]
        self.out_strides = [4 * 2 ** i for i in self.return_idx]
        if freeze_at >= 0:
            self.stem.requires_grad_(False)
            for stage in self.stages.layers[:freeze_at]: stage.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx: outs.append(x)
        return outs

    def load_timm(self, path):
        """Load a timm ``hgnetv2_b*`` ``model.safetensors``; the classifier head is dropped."""
        state = {name: value for name, value in read_safetensors(path).items() if not name.startswith('head.')}
        self.load_state_dict(state)
        return self


for _variant in VARIANTS:
    register(f'hgnetv2_{_variant}')(lambda _variant=_variant, **kwargs: HGNetV2(_variant, **kwargs))
