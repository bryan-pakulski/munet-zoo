"""U-Net: a convolutional encoder, a decoder with skip connections, and a per-pixel class head."""
import munet as mu
from munet import nn


class _Block(nn.Module):
    def __init__(self, cin, cout):
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())

    def forward(self, x):
        return self.body(x)


class UNet(nn.Module):
    """Input (N, in_channels, H, W) with H and W divisible by 2**depth; output logits (N, num_classes, H, W)."""

    def __init__(self, num_classes, in_channels=3, base=16, depth=3):
        if depth < 1:
            raise ValueError('depth is at least 1')
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.depth = depth
        self.down = nn.ModuleList([_Block(i, o) for i, o in zip([in_channels] + widths[:-1], widths)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)
        self.reduce = nn.ModuleList([nn.Conv2d(widths[i + 1], widths[i], 1) for i in range(depth)])
        self.merge = nn.ModuleList([_Block(2 * widths[i], widths[i]) for i in range(depth)])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        for i in reversed(range(self.depth)):
            x = self.merge[i](mu.cat([skips[i], self.reduce[i](self.up(x))], 1))
        return self.head(x)
