"""Training steps, checkpoint/resume, export and post-processing. CPU by default; MUNET_TEST_VULKAN=1 adds the GPU tests."""
import os
import numpy as np
import pytest
import munet as mu

from munet_zoo.rtdetrv2 import RTDETRv2, DetectorTrainer, PostProcessor

vulkan = pytest.mark.skipif(os.environ.get('MUNET_TEST_VULKAN') != '1', reason='set MUNET_TEST_VULKAN=1 to run on the GPU')

IMAGE = np.random.default_rng(5).uniform(size=(2, 3, 64, 64)).astype(np.float32)
TARGETS = [{'labels': [1], 'boxes': [[.32, .48, .17, .21]]}, {'labels': [], 'boxes': np.zeros((0, 4))}]


def small_model():
    return RTDETRv2(3, backbone='convnextv2_atto', hidden_dim=16, nhead=4, num_queries=4, num_decoder_layers=2, num_points=2,
                    dim_feedforward=32, num_denoising=4, expansion=.5, depth_mult=.34)


def test_checkpoint_resumes_training_and_export_matches(tmp_path):
    trainer = DetectorTrainer(small_model(), device='cpu', target_slots=2, milestones=(1,), seed=17)
    first = trainer.step(IMAGE, TARGETS)
    assert np.isfinite(list(first.values())).all()
    trainer.finish_epoch()
    trainer.save(tmp_path / 'state.mnet')
    resumed = DetectorTrainer(small_model(), device='cpu', target_slots=2, milestones=(1,), seed=999).load(tmp_path / 'state.mnet')
    assert resumed.steps == 1 and resumed.epoch == 1
    for _ in range(2):
        expected, actual = trainer.step(IMAGE, TARGETS), resumed.step(IMAGE, TARGETS)
        np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=3e-4, atol=3e-5)
    other = dict(resumed.model.named_parameters())
    for name, p in trainer.model.named_parameters():
        np.testing.assert_allclose(p.numpy(), other[name].numpy(), rtol=2e-4, atol=3e-5, err_msg=name)
    inference = trainer.export(tmp_path / 'detector.mnet', IMAGE[:1])
    loaded = mu.load(tmp_path / 'detector.mnet', device='cpu')
    expected, actual = inference(IMAGE[:1]), loaded(IMAGE[:1])
    for key in expected: np.testing.assert_allclose(actual[key].numpy(), expected[key].numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('options', [{}, {'cross_attn_method': 'discrete'}])
def test_onnx_export_matches_onnxruntime(options, tmp_path):
    ort = pytest.importorskip('onnxruntime')
    from munet.interop import to_onnx
    model = RTDETRv2(3, backbone='convnextv2_atto', hidden_dim=16, nhead=4, num_queries=4, num_decoder_layers=2, num_points=2,
                     dim_feedforward=32, num_denoising=4, expansion=.5, depth_mult=.34, **options).eval()
    # A freshly initialized decoder samples exactly on cell boundaries, where one ULP decides which cell discrete
    # sampling reads and two runtimes may disagree. Perturbed weights move the sampling points off the boundaries.
    rng = np.random.default_rng(3)
    for p in model.decoder.parameters(): p.assign(p.numpy() + rng.normal(size=tuple(p.shape)).astype(np.float32) * 0.02)
    program = mu.compile(model, device='cpu', precision='fp32')
    expected = {k: v.numpy() for k, v in program(IMAGE[:1]).items()}
    to_onnx(program, tmp_path / 'detector.onnx')
    session = ort.InferenceSession(str(tmp_path / 'detector.onnx'), providers=['CPUExecutionProvider'])
    actual = session.run(None, {session.get_inputs()[0].name: IMAGE[:1]})
    assert len(actual) == len(expected)
    for value, (name, want) in zip(actual, expected.items()): np.testing.assert_allclose(value, want, rtol=3e-4, atol=3e-5, err_msg=name)


def test_postprocessor_scales_boxes_to_the_original_size():
    logits = np.full((1, 2, 3), -5, np.float32)
    logits[0, 1, 2] = 5
    boxes = np.array([[[.5, .5, .2, .2], [.25, .5, .5, 1.]]], np.float32)
    sizes = np.array([[200, 100]], np.float32)  # width, height
    post = PostProcessor(1)
    out = mu.compile(lambda l, b, s: post({'pred_logits': l, 'pred_boxes': b}, s), device='cpu')(logits, boxes, sizes)
    assert out['labels'].numpy().tolist() == [[2]]
    np.testing.assert_allclose(out['boxes'].numpy(), [[[0, 0, 100, 100]]], atol=1e-3)


@vulkan
def test_vulkan_inference_matches_cpu():
    model = small_model().eval()
    expected = {k: v.numpy() for k, v in mu.compile(model, device='cpu', precision='fp32')(IMAGE).items()}
    actual = mu.compile(model, device='vulkan', precision='fp32')(IMAGE)
    for key, want in expected.items(): np.testing.assert_allclose(actual[key].numpy(), want, rtol=2e-3, atol=2e-4, err_msg=key)


@vulkan
def test_vulkan_training_step_matches_cpu():
    gpu = DetectorTrainer(small_model(), device='vulkan', target_slots=2, seed=17)
    cpu = DetectorTrainer(small_model(), device='cpu', target_slots=2, seed=17)
    for _ in range(2):
        actual, expected = gpu.step(IMAGE, TARGETS), cpu.step(IMAGE, TARGETS)
        assert set(actual) == set(expected)
        for name in expected: np.testing.assert_allclose(actual[name], expected[name], rtol=3e-2, atol=3e-3, err_msg=name)
