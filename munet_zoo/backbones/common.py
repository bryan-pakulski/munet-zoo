# ConvNormLayer is adapted for MuNet from lyuwenyu/RT-DETR (Apache-2.0); see munet_zoo/rtdetrv2/LICENSE-RT-DETR and NOTICE.
"""Layers shared by backbones and detector necks."""
from munet import nn


def get_activation(act):
    if act is None: return nn.Identity()
    if isinstance(act, nn.Module): return act
    types = {'relu': nn.ReLU, 'silu': nn.SiLU, 'swish': nn.SiLU, 'gelu': nn.GELU, 'sigmoid': nn.Sigmoid}
    if act not in types: raise ValueError(f"unsupported activation: {act}")
    return types[act]()


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size, stride, groups=1, padding=None, act=None):
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride, padding=(filter_size - 1) // 2 if padding is None else padding,
                              groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x): return self.act(self.norm(self.conv(x)))
