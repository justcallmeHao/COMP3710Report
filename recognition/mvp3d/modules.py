import torch
from torch import nn
from torch.nn import functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([block(1, 16), block(16, 32), block(32, 64)])
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = block(64, 128)
        self.reducers = nn.ModuleList([nn.Conv3d(a, b, 3, padding=1)
                                       for a, b in [(128, 64), (64, 32), (32, 16)]])
        self.decoders = nn.ModuleList([block(128, 64), block(64, 32), block(32, 16)])
        self.output = nn.Conv3d(16, 6, 1)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(n % 8 or n < 16 for n in x.shape[2:]):
            raise ValueError("Expected N,1,D,H,W; spatial dimensions >=16 divisible by 8.")
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for reduce, decode, skip in zip(self.reducers, self.decoders, reversed(skips)):
            x = reduce(F.interpolate(x, scale_factor=2, mode="nearest"))
            x = decode(torch.cat((x, skip), dim=1))
        return self.output(x)
