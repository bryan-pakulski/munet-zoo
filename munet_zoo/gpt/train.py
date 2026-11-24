"""Train the GPT on Tiny Shakespeare, one token per character: python -m munet_zoo.gpt.train [--alone | ...]"""
import os

import numpy as np
import munet as mu

from ..training import evaluating, fetch, parser, seeded, train
from . import GPT, lm_loss

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
CONTEXT = 256


def main():
    args = parser(__doc__, data='data/shakespeare', epochs=10, batch=32, shard_rows=8, lr=6e-4).parse_args()
    text = open(fetch(URL, os.path.join(args.data, 'input.txt')), encoding='utf-8').read()
    chars = sorted(set(text))
    ids = np.array([chars.index(c) for c in text], np.int64)
    cut = int(len(ids) * .9)
    train_ids, val_ids = ids[:cut], ids[cut:]
    model = seeded(lambda: GPT(len(chars), context=CONTEXT, dim=384, depth=6, heads=6), args.seed)
    optimizer = mu.optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=0.1)

    def loss_fn(window):
        return lm_loss(model(window), window)

    def epoch(rng, n):
        """Every CONTEXT-long window of the training text once, from a random offset, in random order."""
        starts = rng.permutation(np.arange(rng.integers(CONTEXT), len(train_ids) - CONTEXT, CONTEXT))
        for i in range(0, len(starts) - n + 1, n):
            yield (np.stack([train_ids[s:s + CONTEXT] for s in starts[i:i + n]]),)

    val = np.stack([val_ids[s:s + CONTEXT] for s in range(0, len(val_ids) - CONTEXT, CONTEXT)])
    val = val[:len(val) // 32 * 32]

    def evaluate(model):
        def body(fwd):
            nll = 0.0
            for i in range(0, len(val), 32):
                w = val[i:i + 32]
                logits = fwd(w.astype(np.float32))[:, :-1].astype(np.float64)
                top = logits.max(2, keepdims=True)
                lse = np.log(np.exp(logits - top).sum(2)) + top[..., 0]
                nll += (lse - np.take_along_axis(logits, w[:, 1:, None], 2)[..., 0]).sum()
            return {'loss': float(nll / (len(val) * (CONTEXT - 1)))}
        return evaluating(model, args.device, body)

    steps = (len(train_ids) - CONTEXT) // CONTEXT // args.batch
    train(args, model, optimizer, loss_fn, epoch, steps, evaluate, name='gpt-shakespeare')


if __name__ == '__main__':
    main()
