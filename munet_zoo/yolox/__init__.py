"""YOLOX: CSPDarknet backbone, PAFPN neck and a decoupled anchor-free head with objectness."""
from .model import YOLOX, yolox_config  # noqa: F401
from .loss import YOLOXLoss, pad_targets  # noqa: F401
from .training import YOLOXTrainer, yolox_sgd  # noqa: F401
