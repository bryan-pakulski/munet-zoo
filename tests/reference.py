"""The pinned upstream RT-DETRv2 (PyTorch) for parity tests, loaded without upstream's config registry.

Only two things are replaced in the upstream text: the ``register`` decorator becomes a no-op and the distributed
helpers become their single-process values. Checksums are verified before any file runs. Fetch the files with
``python tools/fetch_reference.py``; tests that need them are skipped when they are absent.
"""
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['box_ops', 'utils', 'common', 'presnet', 'denoising', 'hybrid_encoder', 'rtdetrv2_decoder', 'matcher', 'rtdetrv2_criterion']
PATCHES = {
    'from ...core import register': 'register = lambda *args, **kwargs: (lambda cls: cls)',
    'from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized':
        'get_world_size = lambda: 1\nis_dist_available_and_initialized = lambda: False',
}
_loaded = None


def load_reference():
    global _loaded
    if _loaded is not None: return _loaded
    manifest = json.loads((ROOT / 'tests/rtdetrv2-reference.json').read_text())
    folder = ROOT / 'reference/rtdetrv2'
    if not all((folder / f'{name}.py').is_file() for name in MODULES):
        pytest.skip('upstream reference missing: run python tools/fetch_reference.py')
    package = types.ModuleType('_upstream_rtdetrv2'); package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    for name in MODULES:
        path = folder / f'{name}.py'; data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['files'][path.name]['sha256']: raise RuntimeError(f'upstream checksum mismatch: {path.name}')
        text = data.decode()
        for old, new in PATCHES.items(): text = text.replace(old, new)
        module = types.ModuleType(f'{package.__name__}.{name}'); module.__package__ = package.__name__; module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(text, str(path), 'exec'), module.__dict__)
        setattr(package, name, module)
    _loaded = package
    return package


def model_for(config):
    """The upstream detector for a munet_zoo RTDETRv2 config with a ``presnet*`` backbone."""
    ref = load_reference()
    depth = int(config['backbone'].removeprefix('presnet'))
    backbone_args = {'variant': 'd', 'return_idx': [1, 2, 3], **config['backbone_args']}
    hidden = config['hidden_dim']

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ref.presnet.PResNet(depth, pretrained=False, **backbone_args)
            self.encoder = ref.hybrid_encoder.HybridEncoder(
                in_channels=[512, 1024, 2048] if depth >= 50 else [128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=hidden,
                nhead=config['nhead'], dim_feedforward=config['dim_feedforward'], expansion=config['expansion'],
                depth_mult=config['depth_mult'], eval_spatial_size=config['eval_spatial_size'])
            self.decoder = ref.rtdetrv2_decoder.RTDETRTransformerv2(
                num_classes=config['num_classes'], hidden_dim=hidden, num_queries=config['num_queries'], feat_channels=[hidden] * 3,
                feat_strides=[8, 16, 32], num_levels=3, num_points=config['num_points'], nhead=config['nhead'],
                num_layers=config['num_decoder_layers'], dim_feedforward=config['dim_feedforward'],
                num_denoising=config['num_denoising'], learn_query_content=config['learn_query_content'],
                eval_spatial_size=config['eval_spatial_size'], cross_attn_method=config['cross_attn_method'],
                query_select_method=config['query_select_method'])

        def forward(self, x, targets=None, denoising=None):
            if denoising is None: return self.decoder(self.encoder(self.backbone(x)), targets)
            decoder_module = ref.rtdetrv2_decoder
            original = decoder_module.get_contrastive_denoising_training_group

            def prepared(*args, **kwargs):
                labels = torch.tensor(denoising['labels'], dtype=torch.long)
                slots, groups = denoising['max_gt'], denoising['num_group']
                positive = [torch.cat([torch.arange(len(t['labels'])) + g * 2 * slots for g in range(groups)]) for t in targets]
                meta = {'dn_num_group': groups, 'dn_num_split': [labels.shape[1], config['num_queries']], 'dn_positive_idx': positive}
                return (self.decoder.denoising_class_embed(labels), torch.tensor(denoising['boxes_unact']),
                        torch.tensor(denoising['attn_mask'], dtype=torch.bool), meta)

            # The same prepared noise on both sides, so the comparison does not depend on two random generators.
            decoder_module.get_contrastive_denoising_training_group = prepared
            try: return self.decoder(self.encoder(self.backbone(x)), targets)
            finally: decoder_module.get_contrastive_denoising_training_group = original

    return Detector()


def criterion_for(classes, **options):
    ref = load_reference()
    matcher = ref.matcher.HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}, use_focal_loss=True, alpha=0.25, gamma=2.0)
    return ref.rtdetrv2_criterion.RTDETRCriterionv2(matcher, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}, ['vfl', 'boxes'],
                                                    alpha=0.75, gamma=2.0, num_classes=classes, **options)
