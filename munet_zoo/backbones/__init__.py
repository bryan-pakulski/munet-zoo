"""Backbones, looked up by name so a detector can swap them.

A backbone is a ``munet.nn.Module`` with:

- ``forward(images)``: NCHW float32 images -> list of NCHW feature maps, finest first;
- ``out_channels`` and ``out_strides``: one entry per returned feature map;
- optional ``mean`` and ``std``: per-channel RGB statistics the detector applies to [0, 1] images before the backbone
  (leave them unset for backbones trained on unnormalised images).

Add your own with ``@register("name")`` on a factory that returns such a module.
"""

_REGISTRY = {}


def register(name):
    def wrap(factory):
        if name in _REGISTRY: raise ValueError(f"backbone {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory
    return wrap


def available(): return sorted(_REGISTRY)


def build_backbone(name, **kwargs):
    if name not in _REGISTRY: raise ValueError(f"unknown backbone {name!r}; available: {available()}")
    backbone = _REGISTRY[name](**kwargs)
    if len(backbone.out_channels) != len(backbone.out_strides): raise ValueError("backbone channels and strides differ in length")
    return backbone


from . import convnext, convnextv2, presnet  # noqa: E402,F401  (registers the built-in backbones)
