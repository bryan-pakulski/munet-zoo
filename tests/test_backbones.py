"""Backbones: ConvNeXt V2 against timm, and swapping backbones under the detector."""
import numpy as np
import pytest
import munet as mu
from munet import nn

from munet_zoo.backbones import available, build_backbone, register
from munet_zoo.rtdetrv2 import RTDETRv2, rtdetrv2_adamw

SMALL = dict(hidden_dim=16, nhead=4, num_queries=4, num_decoder_layers=2, num_points=2, dim_feedforward=32,
             num_denoising=4, expansion=.5, depth_mult=.34)


@register('test_tiny')
class TinyBackbone(nn.Module):
    """Three strided convolutions and no input statistics: the smallest thing that meets the backbone contract."""
    def __init__(self, width=8):
        self.layers = nn.ModuleList([nn.Conv2d(3 if i == 0 else width, width, 3, 8 if i == 0 else 2, 1) for i in range(3)])
        self.out_channels, self.out_strides = [width] * 3, [8, 16, 32]

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x).relu()
            outs.append(x)
        return outs


@pytest.mark.parametrize('variant', ['atto', 'tiny'])  # atto stores the block MLP as 1x1 convolutions, tiny as linear layers
def test_convnextv2_matches_timm(variant, tmp_path):
    timm = pytest.importorskip('timm')
    torch = pytest.importorskip('torch')
    save_file = pytest.importorskip('safetensors.torch').save_file
    torch.manual_seed(3)
    reference = timm.create_model(f'convnextv2_{variant}', pretrained=False).eval()
    with torch.no_grad():
        for name, p in reference.named_parameters():
            if '.grn.' in name: p.normal_(std=0.5)  # zero at initialization, which would hide the layer
    save_file(reference.state_dict(), str(tmp_path / 'model.safetensors'))
    model = build_backbone(f'convnextv2_{variant}').load_timm(tmp_path / 'model.safetensors').eval()
    image = np.random.default_rng(4).normal(size=(2, 3, 64, 96)).astype(np.float32)
    with torch.no_grad():
        x, expected = reference.stem(torch.tensor(image)), []
        for stage in reference.stages:
            x = stage(x)
            expected.append(x.numpy())
    result = mu.compile(model, device='cpu', precision='fp32')(image)
    assert model.out_strides == [8, 16, 32] and len(result) == 3
    for actual, want in zip(result, expected[1:]):
        np.testing.assert_allclose(actual.numpy(), want, rtol=5e-4, atol=1e-5 * np.abs(want).max())


@pytest.mark.parametrize('name', ['convnextv2_atto', 'presnet18', 'test_tiny'])
def test_detector_runs_with_any_registered_backbone(name):
    model = RTDETRv2(3, backbone=name, **SMALL).eval()
    assert RTDETRv2(**model.config).config == model.config
    assert (model.mean is not None) == name.startswith('convnextv2')
    image = np.random.default_rng(9).uniform(size=(1, 3, 64, 96)).astype(np.float32)
    out = {k: v.numpy() for k, v in mu.compile(model, device='cpu')(image).items()}
    assert out['pred_logits'].shape == (1, 4, 3) and out['pred_boxes'].shape == (1, 4, 4)
    assert all(np.isfinite(v).all() for v in out.values())


def test_registry_and_backbone_arguments():
    assert {'convnextv2_atto', 'convnextv2_tiny', 'presnet18', 'presnet50', 'test_tiny'} <= set(available())
    with pytest.raises(ValueError): RTDETRv2(3, backbone='no_such_backbone', **SMALL)
    with pytest.raises(ValueError): register('test_tiny')(TinyBackbone)
    model = RTDETRv2(3, backbone='convnextv2_atto', backbone_args={'freeze_at': 4}, **SMALL)
    frozen = [p.requires_grad for n, p in model.named_parameters() if n.startswith('backbone.')]
    assert frozen and not any(frozen)
    assert all(g['lr'] == 1e-4 for g in rtdetrv2_adamw(model).param_groups)  # no backbone group is left to train
    partly = build_backbone('convnextv2_atto', freeze_at=1)
    assert not any(p.requires_grad for p in partly.stem.parameters() + partly.stages[0].parameters())
    assert all(p.requires_grad for p in partly.stages[1].parameters())
