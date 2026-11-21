"""Sequence tagger: embedding, a bidirectional-free LSTM stack, per-step classes (time series or text)."""
from munet import nn
from munet.nn import functional as F


class SeqTagger(nn.Module):
    """Input features (N, T, features); output logits (N, T, num_classes)."""

    def __init__(self, features, num_classes, hidden=32, layers=2):
        self.rnn = nn.LSTM(features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.rnn(x)[0])


def tagging_loss(logits, labels):
    n, t, c = logits.shape
    return F.cross_entropy(logits.reshape(n * t, c), labels.reshape(n * t))
