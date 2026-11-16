"""ConvNeXt V2 (Woo et al., 2023) built from MuNet layers.

Parameter names match timm's ``convnextv2_*`` models, so their weights load with ``load_timm``. Weights are not part of
this repository. The FCMAE weights published by Meta through timm are licensed CC-BY-NC-4.0 (non-commercial).
"""
import json
import numpy as np
from munet import nn
from munet.core import Parameter
from . import register

# variant: (blocks per stage, channels per stage)
VARIANTS = {
    'atto': ((2, 2, 6, 2), (40, 80, 160, 320)),
    'femto': ((2, 2, 6, 2), (48, 96, 192, 384)),
    'pico': ((2, 2, 6, 2), (64, 128, 256, 512)),
    'nano': ((2, 2, 8, 2), (80, 160, 320, 640)),
    'tiny': ((3, 3, 9, 3), (96, 192, 384, 768)),
    'base': ((3, 3, 27, 3), (128, 256, 512, 1024)),
}


def _per_channel(parameter): return parameter.reshape(1, -1, 1, 1)


class LayerNorm2d(nn.Module):
    """Layer normalization over the channels of an NCHW tensor."""
    def __init__(self, channels, eps=1e-6):
        self.eps = eps
        self.weight = Parameter(np.ones(channels, np.float32))
        self.bias = Parameter(np.zeros(channels, np.float32))

    def forward(self, x):
        centered = x - x.mean(1, keepdim=True)
        y = centered / (centered.square().mean(1, keepdim=True) + self.eps).sqrt()
        return y * _per_channel(self.weight) + _per_channel(self.bias)


class GlobalResponseNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        self.eps = eps
        self.weight = Parameter(np.zeros(channels, np.float32))
        self.bias = Parameter(np.zeros(channels, np.float32))

    def forward(self, x):
        norm = (x.square().sum((2, 3), keepdim=True) + 1e-12).sqrt()  # the offset keeps the gradient finite at zero
        scale = norm / (norm.mean(1, keepdim=True) + self.eps)
        return x + _per_channel(self.bias) + _per_channel(self.weight) * (x * scale)


class Mlp(nn.Module):
    def __init__(self, channels):
        self.fc1 = nn.Conv2d(channels, 4 * channels, 1)
        self.grn = GlobalResponseNorm(4 * channels)
        self.fc2 = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x): return self.fc2(self.grn(self.fc1(x).gelu()))


class Block(nn.Module):
    def __init__(self, channels):
        self.conv_dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.mlp = Mlp(channels)

    def forward(self, x): return x + self.mlp(self.norm(self.conv_dw(x)))


class Stage(nn.Module):
    def __init__(self, ch_in, ch_out, depth, downsample):
        self.downsample = nn.Sequential(LayerNorm2d(ch_in), nn.Conv2d(ch_in, ch_out, 2, 2)) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[Block(ch_out) for _ in range(depth)])

    def forward(self, x): return self.blocks(self.downsample(x))


def read_safetensors(path):
    """Tensors of a .safetensors file as float32 NumPy arrays."""
    with open(path, 'rb') as file:
        header = json.loads(file.read(int.from_bytes(file.read(8), 'little')))
        data = file.read()
    dtypes = {'F16': np.float16, 'F32': np.float32, 'F64': np.float64}
    tensors = {}
    for name, entry in header.items():
        if name == '__metadata__': continue
        start, end = entry['data_offsets']
        tensors[name] = np.frombuffer(data[start:end], dtypes[entry['dtype']]).reshape(entry['shape']).astype(np.float32)
    return tensors


class ConvNeXtV2(nn.Module):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, variant='atto', return_idx=(1, 2, 3), freeze_at=-1):
        """return_idx: stages whose output is returned (stage i has stride 4 * 2**i). freeze_at: -1 trains everything,
        n >= 0 freezes the stem and the first n stages."""
        if variant not in VARIANTS: raise ValueError(f"unknown ConvNeXt V2 variant {variant!r}; available: {sorted(VARIANTS)}")
        depths, dims = VARIANTS[variant]
        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 4, 4), LayerNorm2d(dims[0]))
        self.stages = nn.ModuleList([Stage(dims[max(i - 1, 0)], dims[i], depths[i], i > 0) for i in range(4)])
        self.return_idx = list(return_idx)
        self.out_channels = [dims[i] for i in self.return_idx]
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
        """Load a timm ``convnextv2_*`` ``model.safetensors``; the classifier head is dropped."""
        state = {}
        for name, value in read_safetensors(path).items():
            if name.startswith('head.'): continue
            if name.endswith(('.mlp.fc1.weight', '.mlp.fc2.weight')) and value.ndim == 2: value = value[:, :, None, None]
            if '.mlp.grn.' in name: value = value.reshape(-1)
            state[name] = value
        self.load_state_dict(state)
        return self


for _variant in VARIANTS:
    register(f'convnextv2_{_variant}')(lambda _variant=_variant, **kwargs: ConvNeXtV2(_variant, **kwargs))
