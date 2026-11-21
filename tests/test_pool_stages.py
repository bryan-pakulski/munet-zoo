"""RT-DETRv2 cut into pool stages: the stages chained equal the model, and a staged pool step equals a single-machine step."""
import threading

import numpy as np
import pytest
import munet as mu

from munet_zoo.rtdetrv2 import RTDETRv2, RTDETRCriterionv2, pad_targets, pool_stages, prepare_denoising

IMAGE = np.random.default_rng(5).uniform(size=(2, 3, 64, 64)).astype(np.float32)
# Box edges of predictions and targets never coincide: GIoU has a kink there.
TARGETS = [{'labels': [1, 0], 'boxes': [[.223, .319, .137, .217], [.713, .617, .193, .113]]}, {'labels': [2], 'boxes': [[.32, .48, .17, .21]]}]
TOKEN = 'zoo-pool-stage-test-token-0123456789abcdef'
LAYERS = 3
SELECTED = ['backbone.stages.1.blocks.0.conv_dw.weight', 'encoder.input_proj.0.conv.weight', 'decoder.enc_score_head.weight',
            'decoder.query_pos_head.layers.0.weight', 'decoder.decoder.layers.0.cross_attn.sampling_offsets.weight',
            'decoder.decoder.layers.2.linear1.weight', 'decoder.dec_bbox_head.1.layers.2.weight', 'decoder.denoising_class_embed.weight']


def small_model():
    return RTDETRv2(3, backbone='convnextv2_atto', hidden_dim=16, nhead=4, num_queries=4, num_decoder_layers=LAYERS, num_points=2,
                    dim_feedforward=32, num_denoising=4, expansion=.5, depth_mult=.34).train()


def shard():
    padded = pad_targets(TARGETS, 3, 2)
    dn = prepare_denoising(padded, 3, 4, num_denoising=4, rng=np.random.default_rng(6))
    return [IMAGE, padded['labels'], padded['boxes'], padded['valid'], dn['labels'], dn['boxes_unact'], dn['attn_mask']], dn


def whole_loss(model, criterion, dn):
    def loss(images, labels, boxes, valid, dn_labels, dn_boxes, mask):
        denoising = {'labels': dn_labels, 'boxes_unact': dn_boxes, 'attn_mask': mask, 'num_group': dn['num_group'], 'max_gt': dn['max_gt']}
        return criterion.total(criterion(model(images, denoising=denoising), {'labels': labels, 'boxes': boxes, 'valid': valid}))
    return loss


@pytest.mark.parametrize('layers_per_stage', [1, 2, 3])
def test_stages_chained_equal_the_model(layers_per_stage):
    arrays, dn = shard()
    model, reference = small_model(), small_model()
    stages = pool_stages(model, RTDETRCriterionv2(3), dn, layers_per_stage=layers_per_stage)
    assert len(stages) == 3 + -(-LAYERS // layers_per_stage)

    def chained(*args):
        out = ()
        for stage in stages:
            out = stage(*out, *args)
            out = tuple(out) if isinstance(out, (tuple, list)) else (out,)
        named = dict(model.named_parameters())
        return out[0], mu.grad(out[0], [named[n] for n in SELECTED])

    def whole(*args):
        loss = whole_loss(reference, RTDETRCriterionv2(3), dn)(*args)
        named = dict(reference.named_parameters())
        return loss, mu.grad(loss, [named[n] for n in SELECTED])

    loss, grads = mu.compile(chained, device='cpu', precision='fp32')(*arrays)
    want_loss, want_grads = mu.compile(whole, device='cpu', precision='fp32')(*arrays)
    np.testing.assert_allclose(loss.item(), want_loss.item(), rtol=1e-5)
    for name, actual, want in zip(SELECTED, grads, want_grads):
        want = want.numpy()
        assert np.abs(want).max() > 0, name
        np.testing.assert_allclose(actual.numpy(), want, rtol=1e-3, atol=1e-6, err_msg=name)
    state = reference.state_dict()
    for name, value in model.state_dict().items():  # BatchNorm statistics move the same way
        np.testing.assert_allclose(value, state[name], rtol=1e-4, atol=1e-6, err_msg=name)


def test_pool_stages_validates_its_arguments():
    arrays, dn = shard()
    with pytest.raises(ValueError, match='layers_per_stage'):
        pool_stages(small_model(), RTDETRCriterionv2(3), dn, layers_per_stage=LAYERS + 1)
    with pytest.raises(ValueError, match='model.train'):
        pool_stages(small_model().eval(), RTDETRCriterionv2(3), dn)


def test_staged_pool_step_equals_a_single_machine_step():
    pool = pytest.importorskip('munet.pool', reason='needs a MuNet with the training pool')
    arrays, dn = shard()

    reference = small_model()
    reference_optimizer = mu.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.01)
    reference_loss = whole_loss(reference, RTDETRCriterionv2(3), dn)

    @mu.compile(device='cpu', precision='fp32')
    def step(*args):
        reference_optimizer.zero_grad()
        loss = reference_loss(*args)
        loss.backward()
        reference_optimizer.step()
        return loss

    expected = [step(*arrays).item() for _ in range(2)]

    model = small_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    stages = pool_stages(model, RTDETRCriterionv2(3), dn, layers_per_stage=2)
    p = pool.Pool(stages, trainable, mu.optim.SGD(trainable, lr=0.01), token=TOKEN, device='cpu')
    host, port = p.serve(port=0)
    node = pool.Node(f'http://{host}:{port}', TOKEN, device='cpu')
    thread = threading.Thread(target=node.run, daemon=True)
    thread.start()
    try:
        losses = [p.step_shards([arrays], timeout=900) for _ in range(2)]
        status = p.status()
    finally:
        node.stop()
        thread.join(30)
        p.close()

    assert status['stages'] == [[k, k] for k in range(len(stages))]
    assert status['nodes'][0]['tasks'] == 2 * (2 * len(stages) - 1) and status['nodes'][0]['rejected'] == 0
    np.testing.assert_allclose(losses, expected, rtol=1e-4)
    state = reference.state_dict()
    for name, value in model.state_dict().items():
        np.testing.assert_allclose(value, state[name], rtol=1e-3, atol=1e-5, err_msg=name)
