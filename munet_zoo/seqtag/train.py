"""Train the LSTM tagger on English Web Treebank part-of-speech tags: python -m munet_zoo.seqtag.train [...]"""
import os

import numpy as np
import munet as mu
from munet import nn

from ..training import evaluating, fetch, parser, seeded, train
from . import SeqTagger, tagging_loss

URL = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-{}.conllu'
TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
        'SYM', 'VERB', 'X']
PAD, LENGTH = len(TAGS), 48  # PAD labels padding positions; it is never predicted for a real word


class Tagger(nn.Module):
    """Word embedding followed by the LSTM tagger. Word 0 is padding, 1 an unknown word."""

    def __init__(self, vocab):
        self.embed = nn.Embedding(vocab, 128)
        self.tagger = SeqTagger(128, PAD + 1, hidden=256, layers=2)

    def forward(self, ids):
        return self.tagger(self.embed(ids))


def read(path):
    sentences, words = [], []
    for line in open(path, encoding='utf-8'):
        fields = line.rstrip('\n').split('\t')
        if len(fields) == 1:
            if words:
                sentences.append(words)
            words = []
        elif fields[0].isdigit() and fields[3] in TAGS:
            words.append((fields[1].lower(), TAGS.index(fields[3])))
    return [s for s in sentences if len(s) <= LENGTH]


def main():
    args = parser(__doc__, data='data/ewt', epochs=20, batch=64, shard_rows=16, lr=2e-3).parse_args()
    train_s, dev_s = (read(fetch(URL.format(s), os.path.join(args.data, f'{s}.conllu'))) for s in ('train', 'dev'))
    counts = {}
    for s in train_s:
        for w, _ in s:
            counts[w] = counts.get(w, 0) + 1
    vocab = {w: i + 2 for i, w in enumerate(sorted(w for w, c in counts.items() if c >= 2))}

    def encode(sentences):
        x = np.zeros((len(sentences), LENGTH), np.int64)
        y = np.full((len(sentences), LENGTH), PAD, np.int64)
        for i, s in enumerate(sentences):
            for j, (w, t) in enumerate(s):
                x[i, j], y[i, j] = vocab.get(w, 1), t
        return x, y
    x, y = encode(train_s)
    xdev, ydev = encode(dev_s[:len(dev_s) // 50 * 50])
    model = seeded(lambda: Tagger(len(vocab) + 2), args.seed)
    optimizer = mu.optim.AdamW(list(model.parameters()), lr=args.lr)

    def loss_fn(ids, tags):
        return tagging_loss(model(ids), tags)

    def epoch(rng, n):
        order = rng.permutation(len(x))
        for i in range(0, len(order) - n + 1, n):
            yield x[order[i:i + n]], y[order[i:i + n]]

    def evaluate(model):
        def body(fwd):
            hit = total = 0
            for i in range(0, len(xdev), 50):
                pred, real = fwd(xdev[i:i + 50].astype(np.float32)).argmax(2), ydev[i:i + 50] != PAD
                hit += ((pred == ydev[i:i + 50]) & real).sum()
                total += real.sum()
            return {'accuracy': float(hit / total)}
        return evaluating(model, args.device, body)

    train(args, model, optimizer, loss_fn, epoch, len(x) // args.batch, evaluate, name='seqtag-ewt')


if __name__ == '__main__':
    main()
