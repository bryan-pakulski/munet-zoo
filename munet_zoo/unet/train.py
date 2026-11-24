"""Train the U-Net on Oxford-IIIT Pet trimaps (pet, background, border): python -m munet_zoo.unet.train [...]"""
import os

import numpy as np
import munet as mu

from ..training import evaluating, fetch, parser, seeded, train, untar
from . import UNet, segmentation_loss

URL = 'https://thor.robots.ox.ac.uk/pets/{}.tar.gz'
SIZE = 128


def load(data):
    """Images and trimaps resized to SIZE, cached as one npz; every fifth image is held out."""
    cache = os.path.join(data, f'pets-{SIZE}.npz')
    if not os.path.exists(cache):
        from PIL import Image
        for part, marker in (('images', 'images'), ('annotations', 'annotations/trimaps')):
            untar(URL.format(part), os.path.join(data, f'{part}.tar.gz'), data, marker)
        trimaps = os.path.join(data, 'annotations', 'trimaps')
        names = sorted(n[:-4] for n in os.listdir(trimaps) if n.endswith('.png') and not n.startswith('.')
                       and os.path.exists(os.path.join(data, 'images', n[:-4] + '.jpg')))
        xs = np.empty((len(names), SIZE, SIZE, 3), np.uint8)
        ys = np.empty((len(names), SIZE, SIZE), np.uint8)
        for i, n in enumerate(names):
            image = Image.open(os.path.join(data, 'images', n + '.jpg')).convert('RGB')
            xs[i] = np.asarray(image.resize((SIZE, SIZE), Image.BILINEAR))
            ys[i] = np.asarray(Image.open(os.path.join(trimaps, n + '.png')).resize((SIZE, SIZE), Image.NEAREST)) - 1
        np.savez(cache, x=xs, y=ys)
    z = np.load(cache)
    held = np.arange(len(z['x'])) % 5 == 0
    return z['x'][~held], z['y'][~held], z['x'][held], z['y'][held]


def prep(images):
    return ((images.astype(np.float32) / 255 - .5) / .25).transpose(0, 3, 1, 2)


def main():
    args = parser(__doc__, data='data/pets', epochs=30, batch=16, shard_rows=4, lr=1e-3).parse_args()
    x, y, xva, yva = load(args.data)
    xva, yva = xva[:len(xva) // 32 * 32], yva[:len(yva) // 32 * 32]
    model = seeded(lambda: UNet(3, base=32, depth=4), args.seed)
    optimizer = mu.optim.AdamW(list(model.parameters()), lr=args.lr)

    def loss_fn(images, masks):
        return segmentation_loss(model(images), masks)

    def epoch(rng, n):
        order = rng.permutation(len(x))
        for i in range(0, len(order) - n + 1, n):
            a, b = x[order[i:i + n]], y[order[i:i + n]]
            flip = rng.random(n) < .5
            a = np.where(flip[:, None, None, None], a[:, :, ::-1], a)
            b = np.where(flip[:, None, None], b[:, :, ::-1], b)
            yield prep(a), b.astype(np.float32)

    def evaluate(model):
        def body(fwd):
            inter, union, hit = np.zeros(3), np.zeros(3), 0
            for i in range(0, len(xva), 32):
                pred, truth = fwd(prep(xva[i:i + 32])).argmax(1), yva[i:i + 32]
                hit += (pred == truth).sum()
                for c in range(3):
                    inter[c] += ((pred == c) & (truth == c)).sum()
                    union[c] += ((pred == c) | (truth == c)).sum()
            return {'miou': float((inter / union).mean()), 'pixel_accuracy': float(hit / yva.size)}
        return evaluating(model, args.device, body)

    train(args, model, optimizer, loss_fn, epoch, len(x) // args.batch, evaluate, name='unet-pets')


if __name__ == '__main__':
    main()
