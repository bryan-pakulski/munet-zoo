# YOLOX architecture: Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (Apache-2.0). Written for MuNet from the paper's
# design; no upstream source is included. Released under this repository's licence; see NOTICE.
"""YOLOX training: compiled step (forward, SimOTA loss, backward, SGD and EMA in one replay), checkpoints and inference."""
import copy
import numpy as np
from munet import compile, optim
from munet.checkpoint import save_state, load_state
from .loss import YOLOXLoss, pad_targets
from .model import YOLOX


def yolox_sgd(model, lr=0.01, momentum=0.9, weight_decay=5e-4):
    """Nesterov SGD; biases and normalization parameters get no weight decay."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{'params': [p for p in params if len(p.shape) > 1]}, {'params': [p for p in params if len(p.shape) <= 1], 'weight_decay': 0}]
    return optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


class YOLOXTrainer:
    """Targets are padded to ``slots`` boxes per image, so one compiled program serves every box count up to ``slots``."""
    def __init__(self, model, *, device='vulkan', optimizer=None, criterion=None, slots=32, ema=True, seed=0):
        if slots < 1: raise ValueError("slots must be positive")
        self.model, self.device, self.slots = model.train(), device, slots
        self.optimizer = optimizer or yolox_sgd(model)
        self.criterion = criterion or YOLOXLoss(model.num_classes)
        self.ema = optim.ModelEMA(model) if ema else None
        self.rng = np.random.default_rng(seed)
        self.steps, self.program = 0, None

    def _compile(self):
        def step(images, labels, boxes, valid):
            self.optimizer.zero_grad()
            losses = self.criterion(self.model(images), {'labels': labels, 'boxes': boxes, 'valid': valid}, size=tuple(images.shape[-2:]))
            losses['total'].backward()
            self.optimizer.step()
            if self.ema: self.ema.update()
            return losses
        return compile(step, device=self.device)

    def step(self, images, targets):
        self.model.train(); images = np.asarray(images)
        if images.dtype != np.float32 or len(images) != len(targets) or not np.isfinite(images).all():
            raise ValueError("expected finite float32 images and one target per image")
        padded = pad_targets(targets, self.slots)
        if self.program is None: self.program = self._compile()
        result = self.program(images, padded['labels'], padded['boxes'], padded['valid'])
        self.steps += 1
        return {k: v.item() for k, v in result.items()} | {'loss': result['total'].item()}

    def state_dict(self):
        return {'config': self.model.config, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'ema': None if self.ema is None else self.ema.state_dict(), 'rng': copy.deepcopy(self.rng.bit_generator.state),
                'steps': self.steps, 'slots': self.slots}

    def save(self, path): save_state(self.state_dict(), path)

    def load(self, path):
        state = load_state(path)
        if state['config'] != self.model.config or state['slots'] != self.slots: raise ValueError("training configuration mismatch")
        if (state['ema'] is None) != (self.ema is None): raise ValueError("EMA configuration mismatch")
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        if self.ema: self.ema.load_state_dict(state['ema'])
        self.rng.bit_generator.state, self.steps = state['rng'], state['steps']
        return self

    def inference_model(self, use_ema=True):
        model = YOLOX(**self.model.config).eval()
        if use_ema and self.ema: self.ema.copy_to(model)
        else: model.load_state_dict(self.model.state_dict())
        return model
