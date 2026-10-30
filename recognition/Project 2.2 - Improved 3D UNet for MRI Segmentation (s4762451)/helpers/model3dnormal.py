import torch
from torch import nn


def double_conv(in_channels, middle_channels, out_channels, settings):
    """build two unpadded 3D convolutions with batch normalization and ReLU"""
    layers = []
    for before, after in ((in_channels, middle_channels), (middle_channels, out_channels)):
        layers.extend([
            nn.Conv3d(before, after, settings["kernel_size"], stride=1, padding=0,
                      dilation=1, groups=1, bias=settings["bias"]),
            nn.BatchNorm3d(after, eps=settings["norm_epsilon"],
                           affine=settings["norm_affine"], track_running_stats=False),
            nn.ReLU(),
        ])
    return nn.Sequential(*layers)


class UNet3D(nn.Module):
    def __init__(self, settings, in_channels, classes):
        """build the baseline 3D U-Net encoder, decoder, and output layer"""
        super().__init__()
        channels = settings["channels"]
        self.encoders = nn.ModuleList([
            double_conv(before, after // 2, after, settings)
            for before, after in zip([in_channels] + channels[:-1], channels)
        ])
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose3d(channel, channel, kernel_size=2, stride=2,
                               padding=0, output_padding=0, bias=True)
            for channel in reversed(channels[1:])
        ])
        self.decoders = nn.ModuleList([
            double_conv(before + after, after, after, settings)
            for before, after in zip(reversed(channels[1:]), reversed(channels[:-1]))
        ])
        self.output = nn.Conv3d(channels[0], classes, kernel_size=1, stride=1,
                                padding=0, bias=True)

    def forward(self, image):
        """predict voxel class logits using cropped encoder skip connections"""
        skips = []
        for encoder in self.encoders[:-1]:
            image = encoder(image)
            skips.append(image)
            image = self.pool(image)
        image = self.encoders[-1](image)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            image = upsample(image)
            starts = [(source - target) // 2 for source, target in zip(skip.shape[2:], image.shape[2:])]
            region = tuple(slice(start, start + size) for start, size in zip(starts, image.shape[2:]))
            skip = skip[(slice(None), slice(None), *region)]
            image = decoder(torch.cat((image, skip), dim=1))
        return self.output(image)
