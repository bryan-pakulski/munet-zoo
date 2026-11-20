"""YOLOX model: output shapes, the anchor grid, box decoding and a compiled CPU forward."""
import numpy as np
import pytest
import munet as mu

from munet_zoo.yolox import YOLOX, yolox_config

CLASSES = 3


def images(size, batch=2): return np.random.default_rng(0).uniform(size=(batch, 3, size, size)).astype(np.float32)


def run(fn, *arrays): return mu.compile(fn, device='cpu')(*arrays).numpy()


@pytest.mark.parametrize('size', [64, 96])
def test_output_shape_and_anchor_count(size):
    model = YOLOX(CLASSES, 'nano').eval()
    out = run(model, images(size))
    anchors = sum((size // s) ** 2 for s in (8, 16, 32))
    assert out.shape == (2, anchors, 5 + CLASSES) and np.isfinite(out).all()
    assert len(YOLOX.grid(size)[0]) == anchors


def test_initial_objectness_and_class_scores_are_low():
    out = run(YOLOX(CLASSES, 'nano').eval(), images(64))
    assert np.all(1 / (1 + np.exp(-out[..., 4:])) < .2)


def test_grid_values():
    grid, stride = YOLOX.grid((64, 96), strides=(32,))
    assert grid.shape == (6, 2) and stride.shape == (6, 1)
    np.testing.assert_array_equal(grid, [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    assert np.all(stride == 32)
    grid, stride = YOLOX.grid(64)
    assert grid.shape == (64 + 16 + 4, 2) and stride[0, 0] == 8 and stride[64, 0] == 16 and stride[-1, 0] == 32
    assert grid[64 + 16 + 3].tolist() == [1, 1] and grid[8].tolist() == [0, 1]


def test_decode_of_zero_output_is_cell_origin_with_stride_size():
    grid, stride = YOLOX.grid(64)
    boxes = run(lambda x: YOLOX.decode(x, 64), np.zeros((1, len(grid), 8), np.float32))
    np.testing.assert_allclose(boxes[0, :, :2], grid * stride, rtol=1e-6)
    np.testing.assert_allclose(boxes[0, :, 2:], np.repeat(stride, 2, 1), rtol=1e-6)


def test_sizes_and_multipliers():
    assert yolox_config('m') == {'depth': .67, 'width': .75}
    assert yolox_config('s', width=.25) == {'depth': .33, 'width': .25}
    with pytest.raises(ValueError, match='unknown YOLOX size'): yolox_config('xl')
    count = lambda m: sum(int(np.prod(p.shape)) for p in m.parameters())
    assert count(YOLOX(CLASSES, 'nano')) < count(YOLOX(CLASSES, 'tiny')) < count(YOLOX(CLASSES, 's'))


def test_rejects_sizes_not_divisible_by_32():
    with pytest.raises(ValueError, match='divisible by 32'): run(YOLOX(CLASSES, 'nano').eval(), images(72))
