# munet-zoo

Models built from [MuNet](https://pypi.org/project/munet-nn/) layers, with training and inference code.

| Model | Package | Backbones |
| --- | --- | --- |
| RT-DETRv2 (object detection) | `munet_zoo.rtdetrv2` | ConvNeXt (incl. DINOv3), ConvNeXt V2, ResNet-vd |
| YOLOX (object detection) | `munet_zoo.yolox` | CSPDarknet (nano, tiny, s, m, l) |
| U-Net (semantic segmentation) | `munet_zoo.unet` | Convolutional encoder and decoder |
| ViT (image classification) | `munet_zoo.vit` | Patch embedding, pre-norm encoder |
| GPT (language model) | `munet_zoo.gpt` | Causal self-attention blocks |
| Sequence tagger (time series, text) | `munet_zoo.seqtag` | LSTM stack |

No weights are stored here. You supply the weight files; see [Weights and licences](#weights-and-licences).

## Install

```bash
pip install munet-nn
pip install -e ".[data]"        # from a clone of this repository; [eval] adds pycocotools
```

## RT-DETRv2

The detector is a backbone, the hybrid encoder and the deformable decoder. Images are NCHW RGB `float32` in [0, 1],
with height and width divisible by 32. The model applies the backbone's own mean and std.

```python
from munet_zoo.rtdetrv2 import RTDETRv2

model = RTDETRv2(num_classes=80, backbone="convnextv2_atto")
model.backbone.load_timm("model.safetensors")     # timm convnextv2_atto weights
```

In eval mode the model returns `pred_logits` `[B, 300, classes]` and `pred_boxes` `[B, 300, 4]` (normalized
cx, cy, w, h).

### Backbones

A backbone is chosen by name. `backbone_args` go to its constructor.

```python
from munet_zoo.backbones import available
available()   # convnext_tiny / small / base / large, convnextv2_atto / femto / pico / nano / tiny / base, presnet18 / 34 / 50 / 101

RTDETRv2(80, backbone="convnextv2_tiny", backbone_args={"freeze_at": 1})   # freeze the stem and the first stage
RTDETRv2(80, backbone="presnet50")
```

To add one, register a factory that returns a `munet.nn.Module` with this interface:

- `forward(images)` returns a list of NCHW feature maps, finest first;
- `out_channels` and `out_strides` have one entry per feature map;
- optional `mean` and `std` (RGB) are applied to the [0, 1] images before the backbone.

```python
from munet import nn
from munet_zoo.backbones import register

class Tiny(nn.Module):
    out_channels, out_strides = [32, 64, 128], [8, 16, 32]
    def __init__(self):
        self.stem = nn.Conv2d(3, 32, 8, 8)
        self.down1 = nn.Conv2d(32, 64, 2, 2)
        self.down2 = nn.Conv2d(64, 128, 2, 2)
    def forward(self, x):
        a = self.stem(x).relu(); b = self.down1(a).relu()
        return [a, b, self.down2(b).relu()]

register("tiny")(Tiny)
model = RTDETRv2(80, backbone="tiny")
```

The encoder and decoder take their channel counts and strides from the backbone, so nothing else changes.

### Train

Targets are one dict per image: `labels` `[N]` and `boxes` `[N, 4]` as normalized cx, cy, w, h.

```python
import numpy as np
from munet_zoo.rtdetrv2 import DetectorTrainer, CocoDetection

data = CocoDetection("train2017", "annotations/instances_train2017.json", horizontal_flip=0.5)
trainer = DetectorTrainer(model, device="vulkan", target_slots=100, milestones=(60,))
rng = np.random.default_rng(0)
for epoch in range(72):
    for images, targets, _, _ in data.batches(4, rng=rng):
        losses = trainer.step(images, targets)     # dict of floats: loss, grad_norm and each loss term
    trainer.finish_epoch()
    trainer.save("detector-state.mnet")            # trainer.load(path) resumes
```

- One step is forward, loss, backward, gradient clipping (0.1), AdamW and EMA in a single compiled GPU program.
- Defaults follow upstream: lr 1e-4, backbone lr 1e-5, weight decay 1e-4 (none on biases and normalization).
  Pass `optimizer=rtdetrv2_adamw(model, ...)` to change them.
- `target_slots` fixes the number of target slots per image. Without it the program is recompiled whenever the
  largest object count in a batch changes. Images with more objects than `target_slots` are an error.
- `CocoDetection` only resizes and flips. Stronger augmentation belongs in your data pipeline.
- The first step compiles the whole training program. With `convnextv2_atto` at 640 px and batch 2 on an RTX 3060
  that took 7.5 minutes and about 10 GB of host memory; later steps took 0.43 s. Each new image size or batch size
  compiles again, so keep them fixed.

### Train on a pool

`pool_stages` cuts the detector into stage functions for MuNet's training pool (`munet.pool`), so that no node has to
hold the whole model. The pool is not in a MuNet release yet (0.11.1 does not have it).

```python
import munet as mu
from munet import pool
from munet_zoo.rtdetrv2 import RTDETRCriterionv2, pad_targets, pool_stages, prepare_denoising, rtdetrv2_adamw

model.train()
criterion, optimizer = RTDETRCriterionv2(model.num_classes), rtdetrv2_adamw(model)

def shard(images, targets):                      # one task: a few images and their targets
    padded = pad_targets(targets, model.num_classes, 100)
    dn = prepare_denoising(padded, model.num_classes, model.num_queries,
                           num_denoising=model.config["num_denoising"], rng=rng)
    return [images, padded["labels"], padded["boxes"], padded["valid"],
            dn["labels"], dn["boxes_unact"], dn["attn_mask"]], dn

def update():
    mu.optim.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()

shards = [shard(images[i:i + 1], targets[i:i + 1]) for i in range(len(images))]
stages = pool_stages(model, criterion, shards[0][1], layers_per_stage=1)
p = pool.Pool(stages, model.parameters(), update, token=token)
p.serve()
loss = p.step_shards([arrays for arrays, _ in shards])
```

- The stages are the backbone, the encoder, query selection, then the decoder layers `layers_per_stage` at a time.
  The last stage also computes the loss.
- The loss reads every decoder layer, so each decoder stage passes on the logits and boxes of the layers before it.
- A staged step equals a single-machine step up to float rounding (tested on a small model).
- Every shard needs the `num_group` and `max_gt` of the `prepare_denoising` result given to `pool_stages`. `max_gt`
  is the largest object count in the shard. Nothing checks this: a shard with other values gets a wrong loss or
  fails. Images with different object counts therefore cannot be mixed yet.

Measured with `convnextv2_atto`, one 640 px image per shard, fp32. Sizes are MuNet's count of a program's buffers, the
number a node's `--memory` limit is compared with. They were the same on an RTX 3060 and an Arc B580.

| `layers_per_stage` | Programs per shard | Largest program | Largest decoder program | Seconds per step |
|---|---|---|---|---|
| no stages | 1 | 1186.0 MiB | | 5.77 |
| 3 | 9 | 389.9 MiB | 324.2 MiB | 8.10 |
| 2 | 11 | 389.9 MiB | 254.7 MiB | not run |
| 1 | 17 | 389.9 MiB | 182.4 MiB | 10.90 |

- Seconds per step: 16 shards, one RTX 3060 node on the same machine, mean of three steps after two.
- The largest program of every cut is the encoder's backward pass. The backbone's backward pass needs 338.4 MiB.
- At most 9.6 MiB passes between two stages.
- Smaller programs cost speed: each stage but the last runs its forward pass twice, and a shard's tasks run in turn.
- A limit close to the largest program is slow. With `--memory 512M` on two nodes nothing was declined, but a node
  dropped and reloaded programs to stay under it: 83 loads of 17 programs in three steps, 12.78 s per 4-shard step.

### Predict and export

```python
from PIL import Image
from munet_zoo.rtdetrv2 import Predictor

predict = Predictor(trainer.inference_model(), device="vulkan")       # EMA weights
for d in predict([Image.open("photo.jpg")], score_threshold=0.5):
    print(d["labels"], d["scores"], d["boxes"])                       # boxes: x1, y1, x2, y2 in original pixels

trainer.export("detector.mnet", images[:1], onnx_path="detector.onnx")
```

MuNet's default precision is `mixed` (fp16 where the GPU supports it). Confident detections match fp32; boxes of
low-scoring queries can differ. Compile with `precision="fp32"` when you need the exact values.

### Official RT-DETRv2 checkpoints

The ResNet backbones keep upstream's parameter names, so the published PyTorch checkpoints load directly
(PyTorch is needed to read the file):

```python
from munet_zoo.rtdetrv2 import RTDETRv2, load_torch_checkpoint

model = RTDETRv2(80, backbone="presnet18", backbone_args={"freeze_norm": False}, expansion=0.5,
                 num_decoder_layers=3, eval_spatial_size=(640, 640))
load_torch_checkpoint(model, "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth")
```

### Differences from upstream

The port follows `lyuwenyu/RT-DETR` (`rtdetrv2_pytorch`) at commit `e97db5a`. Known differences:

- Discrete sampling (`cross_attn_method="discrete"`) clamps x to the feature width and y to its height. Upstream
  clamps both to the height.
- Discrete sampling reads the nearest cell, so a point exactly on a cell boundary can round either way in another
  runtime. An untrained model samples on boundaries; its ONNX export can differ from MuNet by about 1e-2 in the logits.
- With `query_select_method="agnostic"`, the encoder loss matches and scores every object as class 0.
- A batch with no objects gets no denoising queries.
- Denoising noise is drawn with NumPy, so a run does not reproduce upstream's random sequence.

## YOLOX

Anchor-free detector with a CSPDarknet backbone, a PAFPN neck and a decoupled head with objectness, at strides 8, 16
and 32. The sizes are `nano`, `tiny`, `s`, `m` and `l`. Targets are padded to a fixed number of slots, so one compiled
program trains on images with any number of boxes (SimOTA assignment runs as graph operations).

```python
from munet_zoo.yolox import YOLOX, YOLOXTrainer

model = YOLOX(num_classes=7, size="s")
trainer = YOLOXTrainer(model, device="vulkan", slots=32)
targets = [{"labels": [0, 3], "boxes": [[320, 300, 40, 60], [100, 90, 20, 20]]},     # cx, cy, w, h in pixels
           {"labels": [], "boxes": []}]
losses = trainer.step(images, targets)         # images: [B, 3, H, W] float32
```

- `step` returns floats: `loss` and the terms `iou`, `obj`, `cls`, `total` and `matched` (the matched anchor count).
- The trainer uses Nesterov SGD (weight decay on weights only) and an EMA. `save`, `load` and
  `inference_model()` work as for RT-DETRv2. More than `slots` boxes in an image is an error.
- `YOLOX.decode(outputs, size)` turns the raw output `[B, anchors, 5 + classes]` into boxes (cx, cy, w, h in pixels).

### Train on a pool

The loss is one function of the batch arrays, so `munet.pool.Pool` trains the model without stages. The pool is not
in a MuNet release yet.

```python
import munet as mu
from munet import pool
from munet_zoo.yolox import YOLOXLoss, pad_targets

criterion, params = YOLOXLoss(7), list(model.parameters())

def loss_fn(images, labels, boxes, valid):
    out = model(images)
    return criterion(out, {"labels": labels, "boxes": boxes, "valid": valid}, size=images.shape[-2:])["total"]

p = pool.Pool(loss_fn, params, mu.optim.SGD(params, lr=0.01, momentum=0.9), token=token, shard_rows=2)
p.serve()
padded = pad_targets(targets, 32)
loss = p.step(images, padded["labels"], padded["boxes"], padded["valid"])
```

## Tests

```bash
python tools/fetch_reference.py      # pinned upstream sources, checksum-verified, into reference/
pytest                               # CPU; needs torch, torchvision, scipy. timm adds the ConvNeXt V2 comparison
MUNET_TEST_VULKAN=1 pytest           # also runs the GPU tests
```

The tests compare outputs, losses, gradients and one optimizer step with the upstream PyTorch code on the same
weights and inputs.

The YOLOX tests have no PyTorch reference: they check shapes, finite losses and gradients, that the loss falls on
synthetic scenes, empty targets, a changing box count, checkpoints and training on a pool.

## Weights and licences

- Code: MIT ([LICENSE](LICENSE)). The RT-DETR parts are adapted from Apache-2.0 sources; see [NOTICE](NOTICE).
- ConvNeXt V2 weights from timm (`timm/convnextv2_*`) are **CC-BY-NC-4.0: non-commercial use only**. A model trained
  from them inherits that restriction. For commercial work, train the backbone yourself or pick another one.
- The official RT-DETRv2 checkpoints are Apache-2.0.
