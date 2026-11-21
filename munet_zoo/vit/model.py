"""Vision Transformer classifier: patch embedding, a class token, pre-norm encoder blocks."""
import numpy as np
from munet import nn
import munet as mu


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp):
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    """Input (N, 3, size, size); output logits (N, num_classes)."""

    def __init__(self, num_classes, size=32, patch=8, dim=64, depth=2, heads=4, mlp=128, in_channels=3):
        if size % patch:
            raise ValueError('size must be a multiple of patch')
        tokens = (size // patch) ** 2 + 1
        self.patchify = nn.Conv2d(in_channels, dim, patch, stride=patch)
        self.cls = nn.Parameter(np.zeros((1, 1, dim), np.float32))
        self.pos = nn.Parameter(np.random.default_rng(0).normal(0, .02, (1, tokens, dim)).astype(np.float32))
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patchify(x)
        n, d = x.shape[0], x.shape[1]
        x = x.reshape(n, d, -1).permute(0, 2, 1)
        x = mu.cat([self.cls + x[:, :1] * 0, x], 1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])
