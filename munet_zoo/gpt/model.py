"""A small GPT: token and position embeddings, causal self-attention blocks, tied-free output head."""
import numpy as np
from munet import nn
from munet.nn import functional as F


class _Block(nn.Module):
    def __init__(self, dim, heads):
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        n, t, d = x.shape
        qkv = self.qkv(self.n1(x))
        q, k, v = [qkv[:, :, i * d:(i + 1) * d].reshape(n, t, self.heads, d // self.heads).permute(0, 2, 1, 3) for i in range(3)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).permute(0, 2, 1, 3).reshape(n, t, d)
        x = x + self.proj(a)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    """Input token ids (N, T) with T <= context; output logits (N, T, vocab)."""

    def __init__(self, vocab, context=32, dim=64, depth=2, heads=4):
        self.context = context
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(np.random.default_rng(0).normal(0, .02, (context, dim)).astype(np.float32))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids):
        x = self.tok(ids) + self.pos[:ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def lm_loss(logits, ids):
    """Next-token cross entropy: position t predicts ids[:, t + 1]."""
    n, t, v = logits.shape
    return F.cross_entropy(logits[:, :-1].reshape(n * (t - 1), v), ids[:, 1:].reshape(n * (t - 1)))
