"""GPU training of the UNet, ViT, GPT and sequence tagger at realistic sizes, on tasks each must learn.

Every model is trained on a generated task and judged on held-out data, alone on one GPU and pooled across the
available GPUs. The tests are skipped when MuNet sees no Vulkan device.
"""
import threading
import time

import numpy as np
import pytest
import munet as mu
from munet import nn, pool

from munet_zoo.gpt import GPT, lm_loss
from munet_zoo.seqtag import SeqTagger, tagging_loss
from munet_zoo.unet import UNet, segmentation_loss
from munet_zoo.vit import ViT

try:
    DEVICES = mu.devices()
except Exception:
    DEVICES = []

pytestmark = pytest.mark.skipif(not DEVICES, reason='no Vulkan device')
TOKEN = 'zoo-gpu-model-test-token-0123456789abcdef'
F = nn.functional


def seeded(make):
    token = nn._initialization_rng.set(np.random.default_rng(0))
    try:
        return make()
    finally:
        nn._initialization_rng.reset(token)


# ---- tasks: each returns (model factory, loss, batch generator, held-out score) ----------------------------------

def shapes_batch(rng, n, size=128):
    """Gray noise with red discs (class 1) and blue squares (class 2); the target is the class of every pixel."""
    y, x = np.mgrid[:size, :size]
    images = rng.normal(0.5, 0.15, (n, 3, size, size)).astype(np.float32)
    masks = np.zeros((n, size, size), np.float32)
    for i in range(n):
        for _ in range(rng.integers(1, 4)):
            cy, cx, r = rng.integers(16, size - 16, 2).tolist() + [int(rng.integers(8, 20))]
            if rng.random() < .5:
                where, cls, color = (y - cy) ** 2 + (x - cx) ** 2 < r * r, 1, (0.9, 0.1, 0.1)
            else:
                where, cls, color = (abs(y - cy) < r) & (abs(x - cx) < r), 2, (0.1, 0.1, 0.9)
            masks[i][where] = cls
            for c in range(3):
                images[i, c][where] = color[c] + rng.normal(0, 0.1, int(where.sum()))
    return images, masks


def gratings_batch(rng, n, size=64, classes=8):
    """Sinusoidal gratings whose orientation is the class; random phase and heavy noise."""
    y, x = np.mgrid[:size, :size] / size
    labels = rng.integers(0, classes, n)
    images = np.empty((n, 3, size, size), np.float32)
    for i, c in enumerate(labels):
        theta = np.pi * c / classes
        wave = np.sin(2 * np.pi * 6 * (x * np.cos(theta) + y * np.sin(theta)) + rng.uniform(0, 2 * np.pi))
        images[i] = wave[None] + rng.normal(0, 0.7, (3, size, size))
    return images, labels.astype(np.float32)


PERM = np.random.default_rng(7).permutation(256)


def chain_batch(rng, n, context=64):
    """Token sequences that follow a fixed random permutation: every token determines the next one."""
    ids = np.empty((n, context), np.int64)
    ids[:, 0] = rng.integers(0, 256, n)
    for t in range(1, context):
        ids[:, t] = PERM[ids[:, t - 1]]
    return (ids.astype(np.float32),)


def running_sign_batch(rng, n, steps=64, features=8):
    """Per-step label: is the running sum of feature 0 positive. Needs memory across the sequence."""
    x = rng.normal(size=(n, steps, features)).astype(np.float32)
    return x, (np.cumsum(x[:, :, 0], 1) > 0).astype(np.float32)


def logits_of(model, device, arrays):
    model.eval()
    try:
        return mu.compile(lambda *a: model(*a), device=device, precision='fp32')(*arrays).numpy()
    finally:
        model.train()


def unet_task():
    def score(model, device, rng):
        images, masks = shapes_batch(rng, 16)
        pred = logits_of(model, device, [images]).argmax(1)
        ious = [((pred == c) & (masks == c)).sum() / max(((pred == c) | (masks == c)).sum(), 1) for c in (1, 2)]
        return float(np.mean(ious))
    return dict(make=lambda: UNet(3, base=32, depth=4), loss=lambda m, a, b: segmentation_loss(m(a), b),
                batch=shapes_batch, score=score, lr=2e-3, steps=120, rows=8, floor=.75, exact=False)


def vit_task():
    def score(model, device, rng):
        images, labels = gratings_batch(rng, 64)
        return float((logits_of(model, device, [images]).argmax(1) == labels).mean())
    return dict(make=lambda: ViT(8, size=64, patch=8, dim=192, depth=6, heads=3, mlp=768),
                loss=lambda m, a, b: F.cross_entropy(m(a), b), batch=gratings_batch, score=score, lr=5e-4,
                steps=300, rows=32, floor=.9, exact=True)


def gpt_task():
    def score(model, device, rng):
        ids, = chain_batch(rng, 32)
        pred = logits_of(model, device, [ids]).argmax(2)[:, :-1]
        return float((pred == ids[:, 1:]).mean())
    return dict(make=lambda: GPT(256, context=64, dim=256, depth=4, heads=4), loss=lambda m, a: lm_loss(m(a), a),
                batch=chain_batch, score=score, lr=1e-3, steps=300, rows=32, floor=.95, exact=True)


def seqtag_task():
    def score(model, device, rng):
        x, labels = running_sign_batch(rng, 64)
        return float((logits_of(model, device, [x]).argmax(2) == labels).mean())
    return dict(make=lambda: SeqTagger(8, 2, hidden=128, layers=2), loss=lambda m, a, b: tagging_loss(m(a), b),
                batch=running_sign_batch, score=score, lr=3e-3, steps=250, rows=64, floor=.9, exact=True)


TASKS = {'unet': unet_task, 'vit': vit_task, 'gpt': gpt_task, 'seqtag': seqtag_task}


# ---- training drivers ------------------------------------------------------------------------------------------

def train_alone(task, device):
    model = seeded(task['make'])
    optimizer = mu.optim.AdamW(model.parameters(), lr=task['lr'])

    @mu.compile(device=device, precision='fp32')
    def step(*a):
        optimizer.zero_grad()
        value = task['loss'](model, *a)
        value.backward()
        optimizer.step()
        return value

    rng = np.random.default_rng(1)
    losses = [step(*task['batch'](rng, task['rows'])).item() for _ in range(task['steps'])]
    return model, losses


def train_pooled(task, devices):
    model = seeded(task['make'])
    params = list(model.parameters())
    p = pool.Pool(lambda *a: task['loss'](model, *a), params, mu.optim.AdamW(params, lr=task['lr']), token=TOKEN,
                  shard_rows=task['rows'] // len(devices), device=devices[0])
    host, port = p.serve(port=0)
    nodes = [pool.Node(f'http://{host}:{port}', TOKEN, device=d, allow_http=True) for d in devices]
    threads = [threading.Thread(target=n.run, daemon=True) for n in nodes]
    for t in threads:
        t.start()
    rng = np.random.default_rng(1)
    try:
        losses = [p.step(*task['batch'](rng, task['rows']), timeout=1800) for _ in range(task['steps'])]
    finally:
        for n in nodes:
            n.stop()
        for t in threads:
            t.join(60)
        p.close()
    return model, losses


def report(name, how, losses, quality, started):
    print(f'{name} {how}: loss {losses[0]:.3f} -> {losses[-1]:.3f}, held-out {quality:.3f}, '
          f'{time.time() - started:.0f} s', flush=True)


@pytest.mark.parametrize('name', sorted(TASKS))
def test_model_trains_to_quality_on_one_gpu(name):
    task, started = TASKS[name](), time.time()
    model, losses = train_alone(task, 'vulkan:0')
    quality = task['score'](model, 'vulkan:0', np.random.default_rng(99))
    report(name, 'alone', losses, quality, started)
    assert np.isfinite(losses).all()
    assert quality >= task['floor']


@pytest.mark.parametrize('name', sorted(TASKS))
def test_model_trains_to_quality_pooled_across_gpus(name):
    task, started = TASKS[name](), time.time()
    devices = [f'vulkan:{i}' for i in range(min(2, len(DEVICES)))]
    model, losses = train_pooled(task, devices)
    quality = task['score'](model, devices[0], np.random.default_rng(99))
    report(name, f'pooled x{len(devices)}', losses, quality, started)
    assert np.isfinite(losses).all()
    assert quality >= task['floor']
