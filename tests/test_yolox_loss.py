"""YOLOX loss and trainer: finite loss and gradients, empty batches, one program for any box count, learning, checkpoints, pool."""
import threading

import numpy as np
import munet as mu
from munet import pool

from munet_zoo.yolox import YOLOX, YOLOXLoss, YOLOXTrainer, pad_targets

SIZE, CLASSES, SLOTS = 64, 2, 8
TOKEN = 'zoo-yolox-pool-test-token-0123456789abcdef'


def model(): return YOLOX(CLASSES, 'nano', seed=0)


def scene(rng, count):
    """A plain background with ``count`` filled rectangles; class 0 is bright, class 1 dark."""
    image, boxes, labels = np.full((3, SIZE, SIZE), .5, np.float32), [], []
    for _ in range(count):
        w, h = rng.integers(12, 28, 2); x, y = rng.integers(0, SIZE - w), rng.integers(0, SIZE - h)
        c = int(rng.integers(0, CLASSES)); image[:, y:y + h, x:x + w] = .95 if c == 0 else .05
        boxes.append([x + w / 2, y + h / 2, w, h]); labels.append(c)
    return image, {'labels': labels, 'boxes': boxes}


def batch(rng, counts):
    scenes = [scene(rng, c) for c in counts]
    return np.stack([s[0] for s in scenes]), [s[1] for s in scenes]


def grads(targets, seed=1):
    net, loss = model(), YOLOXLoss(CLASSES)
    images = np.random.default_rng(seed).uniform(size=(2, 3, SIZE, SIZE)).astype(np.float32)
    padded = pad_targets(targets, SLOTS)

    @mu.compile(device='cpu')
    def step(images, labels, boxes, valid):
        net.zero_grad() if hasattr(net, 'zero_grad') else None
        terms = loss(net(images), {'labels': labels, 'boxes': boxes, 'valid': valid}, size=SIZE)
        terms['total'].backward()
        return terms['total'], [p.grad for p in net.parameters() if p.grad is not None]

    total, grad = step(images, padded['labels'], padded['boxes'], padded['valid'])
    return total.item(), [g.numpy() for g in grad]


def test_loss_and_gradients_are_finite_and_nonzero():
    total, grad = grads([{'labels': [0, 1], 'boxes': [[20, 20, 16, 16], [44, 40, 20, 24]]}, {'labels': [1], 'boxes': [[32, 32, 30, 30]]}])
    assert np.isfinite(total) and total > 0
    assert grad and all(np.isfinite(g).all() for g in grad) and any(np.abs(g).max() > 0 for g in grad)


def test_empty_targets_give_finite_loss_and_gradients():
    total, grad = grads([{'labels': [], 'boxes': []}, {'labels': [], 'boxes': []}])
    assert np.isfinite(total) and all(np.isfinite(g).all() for g in grad)


def test_pad_targets_masks_padding():
    padded = pad_targets([{'labels': [1], 'boxes': [[1, 2, 3, 4]]}, {'labels': [], 'boxes': []}], 3)
    assert padded['valid'].tolist() == [[1, 0, 0], [0, 0, 0]] and padded['boxes'][0, 0].tolist() == [1, 2, 3, 4]


def test_trainer_learns_and_one_program_serves_any_box_count(tmp_path):
    rng = np.random.default_rng(0)
    trainer = YOLOXTrainer(model(), device='cpu', slots=SLOTS, ema=True)
    losses = []
    for i in range(30):
        losses.append(trainer.step(*batch(rng, [1 + i % 3, 2])) ['loss'])
    assert np.isfinite(losses).all() and np.mean(losses[-5:]) < np.mean(losses[:5])
    program = trainer.program
    assert np.isfinite(trainer.step(*batch(rng, [0, 5]))['loss']) and trainer.program is program

    path = tmp_path / 'yolox.ckpt'
    trainer.save(path)
    probe = batch(np.random.default_rng(9), [2, 3])
    twin = YOLOXTrainer(model(), device='cpu', slots=SLOTS, ema=True).load(path)
    assert twin.steps == trainer.steps
    np.testing.assert_allclose(twin.step(*probe)['loss'], trainer.step(*probe)['loss'], rtol=1e-4)
    assert twin.inference_model().eval() is not None


def test_pool_trains_with_one_loss_function():
    net, loss = model(), YOLOXLoss(CLASSES)
    trainable = list(net.parameters())

    def loss_fn(images, labels, boxes, valid):
        return loss(net(images), {'labels': labels, 'boxes': boxes, 'valid': valid}, size=SIZE)['total']

    p = pool.Pool(loss_fn, trainable, mu.optim.SGD(trainable, lr=0.01, momentum=0.9), token=TOKEN, shard_rows=2, device='cpu')
    host, port = p.serve(port=0)
    node = pool.Node(f'http://{host}:{port}', TOKEN, device='cpu')
    thread = threading.Thread(target=node.run, daemon=True); thread.start()
    images, targets = batch(np.random.default_rng(3), [2, 1, 3, 0])
    padded = pad_targets(targets, SLOTS)
    try:
        losses = [p.step(images, padded['labels'], padded['boxes'], padded['valid'], timeout=600) for _ in range(3)]
    finally:
        node.stop(); thread.join(30); p.close()
    assert np.isfinite(losses).all()
