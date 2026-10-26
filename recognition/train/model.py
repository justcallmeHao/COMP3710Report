import torch
from torch import nn


def spatial_tuple(value):
    return (value,) * 3 if isinstance(value, int) else tuple(value)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, settings, stride=1):
        super().__init__()
        kernel = spatial_tuple(settings["kernel_size"])
        stride = spatial_tuple(stride)
        padding = tuple(k // 2 for k in kernel)
        norm = {"eps": settings["norm_epsilon"], "affine": settings["norm_affine"]}
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, stride, padding, bias=settings["bias"]),
            nn.InstanceNorm3d(out_channels, **norm),
            nn.LeakyReLU(settings["negative_slope"]),
            nn.Dropout3d(settings["dropout"]),
            nn.Conv3d(out_channels, out_channels, kernel, padding=padding, bias=settings["bias"]),
            nn.InstanceNorm3d(out_channels, **norm),
        )
        self.shortcut = (nn.Identity() if in_channels == out_channels and stride == (1, 1, 1)
                         else nn.Conv3d(in_channels, out_channels, 1, stride, bias=settings["bias"]))
        self.activation = nn.LeakyReLU(settings["negative_slope"])

    def forward(self, image):
        return self.activation(self.main(image) + self.shortcut(image))


class ResidualUNet3D(nn.Module):
    def __init__(self, settings, in_channels, classes):
        super().__init__()
        channels, strides = settings["channels"], settings["strides"]
        self.encoders = nn.ModuleList([
            ResidualBlock(in_channels, channels[0], settings),
            *[ResidualBlock(before, after, settings, stride)
              for before, after, stride in zip(channels[:-1], channels[1:], strides)],
        ])
        kernel = spatial_tuple(settings["up_kernel_size"])
        self.upsamples, self.decoders = nn.ModuleList(), nn.ModuleList()
        for before, after, stride in zip(reversed(channels[1:]), reversed(channels[:-1]), reversed(strides)):
            stride = spatial_tuple(stride)
            self.upsamples.append(nn.ConvTranspose3d(
                before, after, kernel, stride, padding=tuple(k // 2 for k in kernel),
                output_padding=tuple(s - 1 for s in stride), bias=settings["bias"]))
            self.decoders.append(ResidualBlock(2 * after, after, settings))
        self.output = nn.Conv3d(channels[0], classes, 1, bias=settings["bias"])

    def forward(self, image):
        skips = []
        for encoder in self.encoders:
            image = encoder(image)
            skips.append(image)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            image = decoder(torch.cat((upsample(image), skip), dim=1))
        return self.output(image)
