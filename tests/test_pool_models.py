"""Pooled training of every zoo family: with no BatchNorm a pooled step equals a single-machine step, and each one learns."""
import threading

import numpy as np
import pytest
import munet as mu
from munet import nn, pool

from munet_zoo.gpt import GPT, lm_loss
from munet_zoo.seqtag import SeqTagger, tagging_loss
from munet_zoo.unet import UNet, segmentation_loss
from munet_zoo.vit import ViT

TOKEN = 'zoo-pool-model-test-token-0123456789abcdef'
ROWS, SHARD = 8, 2
rng = np.random.default_rng(1)


def seeded(make):
    token = nn._initialization_rng.set(np.random.default_rng(0))
    try:
        return make()
    finally:
        nn._initialization_rng.reset(token)


def vit_case():
    x = rng.normal(size=(ROWS, 3, 16, 16)).astype(np.float32)
    y = (x.mean((1, 2, 3)) > 0).astype(np.float32)
    return lambda: ViT(2, size=16, patch=8, dim=16, depth=1, heads=2, mlp=32), \
        lambda m, a, b: nn.functional.cross_entropy(m(a), b), [x, y]


def gpt_case():
    ids = (np.arange(ROWS * 12).reshape(ROWS, 12) % 5).astype(np.float32)
    return lambda: GPT(8, context=12, dim=16, depth=1, heads=2), lambda m, a: lm_loss(m(a), a), [ids]


def seqtag_case():
    x = rng.normal(size=(ROWS, 6, 3)).astype(np.float32)
    y = (x[:, :, 0] > 0).astype(np.float32)
    return lambda: SeqTagger(3, 2, hidden=8, layers=1), lambda m, a, b: tagging_loss(m(a), b), [x, y]


def unet_case():
    x = rng.normal(size=(ROWS, 3, 16, 16)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    return lambda: UNet(2, base=4, depth=2), lambda m, a, b: segmentation_loss(m(a), b), [x, y]


EXACT = {'vit': vit_case, 'gpt': gpt_case, 'seqtag': seqtag_case}


def run_pool(make, loss, arrays, steps, lr):
    model = seeded(make)
    params = list(model.parameters())
    p = pool.Pool(lambda *a: loss(model, *a), params, mu.optim.SGD(params, lr=lr), token=TOKEN, shard_rows=SHARD, device='cpu')
    host, port = p.serve(port=0)
    node = pool.Node(f'http://{host}:{port}', TOKEN, device='cpu')
    thread = threading.Thread(target=node.run, daemon=True); thread.start()
    try:
        losses = [p.step(*arrays, timeout=600) for _ in range(steps)]
    finally:
        node.stop(); thread.join(30); p.close()
    return losses, model.state_dict()


def run_single(make, loss, arrays, steps, lr):
    model = seeded(make)
    optimizer = mu.optim.SGD(model.parameters(), lr=lr)

    @mu.compile(device='cpu', precision='fp32')
    def step(*a):
        optimizer.zero_grad()
        value = loss(model, *a)
        value.backward()
        optimizer.step()
        return value

    return [step(*arrays).item() for _ in range(steps)], model.state_dict()


@pytest.mark.parametrize('name', sorted(EXACT))
def test_pooled_steps_equal_single_machine_steps(name):
    make, loss, arrays = EXACT[name]()
    got, state = run_pool(make, loss, arrays, 3, 0.05)
    want, ref = run_single(make, loss, arrays, 3, 0.05)
    np.testing.assert_allclose(got, want, rtol=1e-4)
    for key, value in ref.items():
        np.testing.assert_allclose(state[key], value, rtol=1e-3, atol=1e-5, err_msg=key)


@pytest.mark.parametrize('case', [vit_case, gpt_case, seqtag_case, unet_case], ids=['vit', 'gpt', 'seqtag', 'unet'])
def test_pooled_training_learns(case):
    make, loss, arrays = case()
    losses, _ = run_pool(make, loss, arrays, 12, 0.1)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
