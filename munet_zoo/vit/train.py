"""Train the ViT on CIFAR-10: python -m munet_zoo.vit.train [--alone | --nodes vulkan:0,vulkan:1 ...]"""
import os
import pickle

import numpy as np
import munet as mu
from munet import nn

from ..training import evaluating, fetch, parser, seeded, train, untar
from . import ViT

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MEAN = np.array([.4914, .4822, .4465], np.float32).reshape(1, 3, 1, 1) * 255
STD = np.array([.247, .243, .261], np.float32).reshape(1, 3, 1, 1) * 255


def load(data):
    untar(URL, os.path.join(data, 'cifar-10-python.tar.gz'), data, 'cifar-10-batches-py/test_batch')

    def read(name):
        with open(os.path.join(data, 'cifar-10-batches-py', name), 'rb') as f:
            b = pickle.load(f, encoding='bytes')
        return b[b'data'].reshape(-1, 3, 32, 32), np.array(b[b'labels'])
    parts = [read(f'data_batch_{i}') for i in range(1, 6)]
    x, y = np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])
    return x, y, *read('test_batch')


def main():
    args = parser(__doc__, data='data/cifar10', epochs=20, batch=128, shard_rows=32, lr=5e-4).parse_args()
    x, y, xte, yte = load(args.data)
    model = seeded(lambda: ViT(10, size=32, patch=4, dim=384, depth=6, heads=6, mlp=1536), args.seed)
    optimizer = mu.optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=0.05)

    def loss_fn(images, labels):
        return nn.functional.cross_entropy(model(images), labels)

    def epoch(rng, n):
        """Random 32 px crops of the 4 px reflect-padded images, flipped half the time."""
        order = rng.permutation(len(x))
        for i in range(0, len(order) - n + 1, n):
            idx = order[i:i + n]
            padded = np.pad(x[idx], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='reflect')
            out = np.empty((n, 3, 32, 32), np.float32)
            for j, (dy, dx) in enumerate(rng.integers(0, 9, (n, 2))):
                crop = padded[j, :, dy:dy + 32, dx:dx + 32]
                out[j] = crop[:, :, ::-1] if rng.random() < .5 else crop
            yield (out - MEAN) / STD, y[idx].astype(np.float32)

    def evaluate(model):
        def body(fwd):
            hit = sum((fwd((xte[i:i + 250].astype(np.float32) - MEAN) / STD).argmax(1) == yte[i:i + 250]).sum()
                      for i in range(0, len(xte), 250))
            return {'accuracy': float(hit / len(xte))}
        return evaluating(model, args.device, body)

    train(args, model, optimizer, loss_fn, epoch, len(x) // args.batch, evaluate, name='vit-cifar10')


if __name__ == '__main__':
    main()
