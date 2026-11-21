"""Output shapes and loss values of the UNet, ViT, GPT and sequence tagger."""
import numpy as np
import pytest
import munet as mu

from munet_zoo.gpt import GPT, lm_loss
from munet_zoo.seqtag import SeqTagger, tagging_loss
from munet_zoo.unet import UNet, segmentation_loss
from munet_zoo.vit import ViT

rng = np.random.default_rng(0)


def run(fn, *arrays):
    return mu.compile(fn, device='cpu', precision='fp32')(*arrays)


def test_unet_outputs_a_class_map_per_pixel():
    x = rng.normal(size=(2, 3, 16, 16)).astype(np.float32)
    mask = rng.integers(0, 3, (2, 16, 16)).astype(np.float32)
    model = UNet(3, base=4, depth=2)
    logits, loss = run(lambda a, m: (model(a), segmentation_loss(model(a), m)), x, mask)
    assert logits.shape == (2, 3, 16, 16)
    assert np.isfinite(loss.item())


def test_unet_rejects_a_depth_below_one():
    with pytest.raises(ValueError, match='depth'):
        UNet(2, depth=0)


def test_vit_outputs_one_logit_row_per_image():
    model = ViT(5, size=16, patch=8, dim=16, depth=1, heads=2, mlp=32)
    assert run(model, rng.normal(size=(2, 3, 16, 16)).astype(np.float32)).shape == (2, 5)
    with pytest.raises(ValueError, match='patch'):
        ViT(5, size=15, patch=8)


def test_gpt_loss_starts_near_log_vocab():
    ids = rng.integers(0, 8, (4, 12)).astype(np.float32)
    model = GPT(8, context=12, dim=16, depth=1, heads=2)
    logits, loss = run(lambda i: (model(i), lm_loss(model(i), i)), ids)
    assert logits.shape == (4, 12, 8)
    assert abs(loss.item() - np.log(8)) < 0.5


def test_gpt_is_causal():
    model = GPT(8, context=12, dim=16, depth=1, heads=2)
    a = rng.integers(0, 8, (1, 12)).astype(np.float32)
    b = a.copy()
    b[0, 8:] = (b[0, 8:] + 1) % 8
    np.testing.assert_allclose(run(model, a).numpy()[:, :8], run(model, b).numpy()[:, :8], atol=1e-5)


def test_seqtag_outputs_classes_per_step():
    x = rng.normal(size=(2, 6, 3)).astype(np.float32)
    y = rng.integers(0, 4, (2, 6)).astype(np.float32)
    model = SeqTagger(3, 4, hidden=8, layers=1)
    logits, loss = run(lambda a, b: (model(a), tagging_loss(model(a), b)), x, y)
    assert logits.shape == (2, 6, 4)
    assert np.isfinite(loss.item())
