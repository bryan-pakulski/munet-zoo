"""RT-DETRv2 parity with the pinned upstream PyTorch implementation (CPU, fp32)."""
import numpy as np
import pytest
import torch
import munet as mu

from munet_zoo.rtdetrv2 import RTDETRv2, RTDETRCriterionv2, pad_targets, prepare_denoising
from munet_zoo.rtdetrv2.training import upstream_state
from munet_zoo.rtdetrv2.utils import deformable_attention_core_func_v2
from .reference import model_for, criterion_for, load_reference


def torch_targets(targets):
    return [{'labels': torch.tensor(t['labels'], dtype=torch.long), 'boxes': torch.tensor(np.asarray(t['boxes'], np.float32)).reshape(-1, 4)}
            for t in targets]


def targets_example():
    # Box edges of predictions and targets never coincide: GIoU has a kink there, where one ULP picks another derivative.
    return [{'labels': [1, 0], 'boxes': [[.223, .319, .137, .217], [.713, .617, .193, .113]]}, {'labels': [], 'boxes': np.zeros((0, 4))}]


@pytest.mark.parametrize('weight_format', [None, 'iou', 'giou'])
def test_criterion_losses_and_gradients(weight_format):
    rng = np.random.default_rng(56); raw = targets_example(); padded = pad_targets(raw, 3)
    logits = rng.normal(size=(2, 6, 3)).astype(np.float32)
    boxes = rng.uniform(.05, .8, size=(2, 6, 4)).astype(np.float32)
    reference = criterion_for(3, boxes_weight_format=weight_format)
    tl = torch.tensor(logits, requires_grad=True); tb = torch.tensor(boxes, requires_grad=True)
    expected = reference({'pred_logits': tl, 'pred_boxes': tb}, torch_targets(raw)); sum(expected.values()).backward()
    criterion = RTDETRCriterionv2(3, boxes_weight_format=weight_format)

    def function(l, b, labels, truth, valid):
        losses = criterion({'pred_logits': l, 'pred_boxes': b}, {'labels': labels, 'boxes': truth, 'valid': valid})
        return losses, mu.grad(criterion.total(losses), [l, b])

    losses, grads = mu.compile(function, device='cpu')(logits, boxes, padded['labels'], padded['boxes'], padded['valid'])
    assert set(losses) == set(expected)
    for name, value in expected.items(): np.testing.assert_allclose(losses[name].item(), value.item(), rtol=4e-5, atol=1e-5, err_msg=name)
    np.testing.assert_allclose(grads[0].numpy(), tl.grad.numpy(), rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(grads[1].numpy(), tb.grad.numpy(), rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize('method', ['default', 'discrete'])
def test_deformable_attention(method):
    ref = load_reference().utils.deformable_attention_core_func_v2
    # Upstream's discrete sampling clamps x with the height, so only square levels are comparable for it.
    shapes = [(2, 3), (1, 2), (1, 1)] if method == 'default' else [(3, 3), (2, 2), (1, 1)]
    points = [3, 2, 1]; cells = sum(h * w for h, w in shapes)
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size=(2, cells, 2, 3)), rng.uniform(-.2, 1.2, size=(2, 4, 2, sum(points), 2)), rng.uniform(size=(2, 4, 2, sum(points)))]
    arrays = [a.astype(np.float32) for a in arrays]
    ts = [torch.tensor(a, requires_grad=True) for a in arrays]
    expected = ref(ts[0], shapes, ts[1], ts[2], points, method=method); expected.square().sum().backward()
    wanted = [0, 1, 2] if method == 'default' else [0, 2]  # discrete sampling has no gradient to the locations

    def function(v, loc, w):
        result = deformable_attention_core_func_v2(v, shapes, loc, w, points, method)
        return result, mu.grad(result.square().sum(), [(v, loc, w)[i] for i in wanted])

    result, grads = mu.compile(function, device='cpu')(*arrays)
    np.testing.assert_allclose(result.numpy(), expected.detach().numpy(), rtol=1e-4, atol=1e-5)
    for actual, i in zip(grads, wanted): np.testing.assert_allclose(actual.numpy(), ts[i].grad.numpy(), rtol=3e-4, atol=3e-5)


def load_from(reference, model):
    """Random upstream weights, with the decoder's zero-initialized parameters perturbed so every path carries signal."""
    with torch.no_grad():
        for p in reference.decoder.parameters(): p.add_(torch.randn_like(p) * 0.02)
    model.load_state_dict(upstream_state(reference.state_dict()))


@pytest.mark.parametrize('options,size', [
    (dict(num_classes=80, backbone='presnet50', num_queries=4, num_denoising=0), (32, 64)),
    (dict(num_classes=3, backbone='presnet18', hidden_dim=32, nhead=4, num_queries=5, num_decoder_layers=3, num_points=[4, 2, 1],
          dim_feedforward=64, num_denoising=0, cross_attn_method='discrete', query_select_method='one2many'), (64, 64)),
    (dict(num_classes=3, backbone='presnet18', hidden_dim=32, nhead=4, num_queries=5, num_decoder_layers=2, dim_feedforward=64,
          num_denoising=0, learn_query_content=True, eval_spatial_size=(64, 96)), (64, 96)),
])
def test_inference(options, size):
    torch.manual_seed(7)
    model = RTDETRv2(**options).eval()
    reference = model_for(model.config).eval(); load_from(reference, model)
    image = np.random.default_rng(77).uniform(size=(1, 3, *size)).astype(np.float32)
    with torch.no_grad():
        backbone = reference.backbone(torch.tensor(image)); encoder = reference.encoder(backbone); output = reference.decoder(encoder)

    def forward(x):
        f = model.backbone(x); e = model.encoder(f)
        return {'backbone': f, 'encoder': e, 'prediction': model.decoder(e)}

    result = mu.compile(forward, device='cpu', fuse=False, precision='fp32')(image)
    for actual, want in zip(result['backbone'], backbone): np.testing.assert_allclose(actual.numpy(), want.numpy(), rtol=3e-4, atol=3e-5)
    for actual, want in zip(result['encoder'], encoder): np.testing.assert_allclose(actual.numpy(), want.numpy(), rtol=8e-4, atol=6e-5)
    assert set(result['prediction']) == set(output)
    for name, want in output.items(): np.testing.assert_allclose(result['prediction'][name].numpy(), want.numpy(), rtol=5e-4, atol=8e-5, err_msg=name)


def test_training_losses_gradients_and_batchnorm_state():
    torch.manual_seed(17)
    # The full ResNet-50-vd with a narrow head, which keeps this gate short.
    model = RTDETRv2(3, backbone='presnet50', backbone_args={'freeze_at': 0}, hidden_dim=16, nhead=4, num_queries=4,
                     num_decoder_layers=2, num_points=[2, 2, 1], dim_feedforward=32, num_denoising=4, expansion=.5, depth_mult=.34)
    reference = model_for(model.config).train(); load_from(reference, model)
    raw = targets_example(); targets = torch_targets(raw); padded = pad_targets(raw, 3)
    dn = prepare_denoising(padded, 3, 4, num_denoising=4, rng=np.random.default_rng(6))
    image = np.random.default_rng(8).uniform(size=(2, 3, 64, 64)).astype(np.float32)
    criterion = RTDETRCriterionv2(3); ref_criterion = criterion_for(3)
    expected = ref_criterion(reference(torch.tensor(image), targets, denoising=dn), targets)
    sum(expected.values()).backward()
    selected = ['backbone.res_layers.0.blocks.0.branch2a.conv.weight', 'encoder.input_proj.0.conv.weight',
                'decoder.decoder.layers.0.cross_attn.sampling_offsets.weight', 'decoder.dec_bbox_head.1.layers.2.weight',
                'decoder.enc_score_head.weight', 'decoder.denoising_class_embed.weight']
    named = dict(model.named_parameters())

    def step(x, labels, boxes, valid, dn_labels, dn_boxes, mask):
        denoising = {'labels': dn_labels, 'boxes_unact': dn_boxes, 'attn_mask': mask, 'num_group': dn['num_group'], 'max_gt': dn['max_gt']}
        losses = criterion(model(x, denoising=denoising), {'labels': labels, 'boxes': boxes, 'valid': valid})
        return losses, mu.grad(criterion.total(losses), [named[n] for n in selected])

    program = mu.compile(step, device='cpu', fuse=False, precision='fp32')
    losses, grads = program(image, padded['labels'], padded['boxes'], padded['valid'], dn['labels'], dn['boxes_unact'], dn['attn_mask'])
    assert set(losses) == set(expected)
    assert any('_enc_' in name for name in losses) and any('_dn_' in name for name in losses) and any('_aux_' in name for name in losses)
    for name, want in expected.items(): np.testing.assert_allclose(losses[name].item(), want.item(), rtol=8e-4, atol=2e-4, err_msg=name)
    ref_params = dict(reference.named_parameters())
    for name, actual in zip(selected, grads):
        want = ref_params[name].grad.numpy()
        assert np.abs(want).max() > 0, name
        np.testing.assert_allclose(actual.numpy(), want, rtol=3e-3, atol=5e-4, err_msg=name)
    state = reference.state_dict()
    for name, value in model.named_buffers():
        if name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            np.testing.assert_allclose(value.numpy(), state[name].numpy(), rtol=1e-3, atol=1e-4, err_msg=name)


def test_empty_ground_truth():
    p = pad_targets([{'labels': [], 'boxes': np.zeros((0, 4))}], 3)
    assert prepare_denoising(p, 3, 4, rng=np.random.default_rng(1)) is None
    c = RTDETRCriterionv2(3)

    def f(l, b, labels, boxes, valid):
        losses = c({'pred_logits': l, 'pred_boxes': b}, {'labels': labels, 'boxes': boxes, 'valid': valid})
        return losses, mu.grad(c.total(losses), [l, b])

    losses, grads = mu.compile(f, device='cpu')(np.zeros((1, 4, 3), np.float32), np.full((1, 4, 4), .2, np.float32), p['labels'], p['boxes'], p['valid'])
    assert losses['loss_bbox'].item() == 0 and losses['loss_giou'].item() == 0
    assert np.all(grads[1].numpy() == 0) and np.isfinite(grads[0].numpy()).all()


def test_training_steps_match_upstream():
    """Forward, loss, backward, gradient clipping and AdamW through DetectorTrainer against PyTorch, on the same batch and
    the same denoising noise.

    Every step starts from PyTorch's weights. Free-running runs are not compared: AdamW moves an element whose gradient is
    near zero by up to a full learning rate in the direction of that gradient's sign, so rounding differences grow until a
    matching flips. A slightly perturbed copy of the PyTorch model leaves the original the same way.
    """
    from munet_zoo.rtdetrv2 import DetectorTrainer
    torch.manual_seed(11)
    model = RTDETRv2(3, backbone='presnet18', hidden_dim=16, nhead=4, num_queries=4, num_decoder_layers=2, num_points=2,
                     dim_feedforward=32, num_denoising=4, expansion=.5, depth_mult=.34)
    reference = model_for(model.config).train(); load_from(reference, model)
    groups = {}
    for name, p in reference.named_parameters():
        if p.requires_grad: groups.setdefault((name.startswith('backbone.'), p.ndim > 1), []).append(p)
    optimizer = torch.optim.AdamW([{'params': params, 'lr': 1e-5 if backbone else 1e-4, 'weight_decay': 1e-4 if decay else 0.0}
                                   for (backbone, decay), params in groups.items()])
    trainable = [p for p in reference.parameters() if p.requires_grad]
    ref_criterion = criterion_for(3)
    raw = targets_example(); targets = torch_targets(raw)
    image = np.random.default_rng(8).uniform(size=(2, 3, 64, 64)).astype(np.float32)
    trainer = DetectorTrainer(model, device='cpu', target_slots=2, seed=17, ema=False)
    rng = np.random.default_rng(17)  # the trainer draws its denoising noise from the same seed, in the same order
    for step in range(4):
        if step: model.load_state_dict(upstream_state(reference.state_dict()))
        dn = prepare_denoising(pad_targets(raw, 3, 2), 3, 4, num_denoising=4, rng=rng)
        optimizer.zero_grad()
        expected = sum(ref_criterion(reference(torch.tensor(image), targets, denoising=dn), targets).values())
        expected.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 0.1)
        optimizer.step()
        actual = trainer.step(image, raw)
        np.testing.assert_allclose(actual['loss'], expected.item(), rtol=1e-4, err_msg=f'loss at step {step}')
        np.testing.assert_allclose(actual['grad_norm'], norm.item(), rtol=1e-3, err_msg=f'gradient norm at step {step}')
        if step: continue
        # One optimizer update. Elements with a clipped gradient near AdamW's eps are left out: their update is the
        # sign of a number that is rounding noise.
        updated = dict(model.named_parameters()); compared = 0
        for name, p in reference.named_parameters():
            if not p.requires_grad: continue
            clear = p.grad.abs().numpy() > 1e-6
            lr = 1e-5 if name.startswith('backbone.') else 1e-4
            difference = np.abs(updated[name].numpy() - p.detach().numpy())[clear]
            compared += difference.size
            if difference.size: assert difference.max() < 0.01 * lr, name
        assert compared > 1000
