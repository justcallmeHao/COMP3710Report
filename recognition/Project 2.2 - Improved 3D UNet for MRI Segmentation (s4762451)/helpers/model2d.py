import torch
from torch import nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=0, dilation=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=0, dilation=1, bias=True),
        nn.ReLU(),
    )


class UNet2D(nn.Module):
    def __init__(self, settings, in_channels, classes):
        super().__init__()
        channels = settings["channels"]
        self.input_patch_size = tuple(settings["input_patch_size"])
        self.output_patch_size = tuple(settings["output_patch_size"])
        self.encoders = nn.ModuleList([
            double_conv(before, after)
            for before, after in zip([in_channels] + channels[:-1], channels)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose2d(before, after, kernel_size=2, stride=2,
                               padding=0, output_padding=0, bias=True)
            for before, after in zip(reversed(channels[1:]), reversed(channels[:-1]))
        ])
        self.decoders = nn.ModuleList([
            double_conv(2 * channel, channel) for channel in reversed(channels[:-1])
        ])
        self.output = nn.Conv2d(channels[0], classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, image):
        skips = []
        for encoder in self.encoders[:-1]:
            image = encoder(image)
            skips.append(image)
            image = self.pool(image)
        image = self.encoders[-1](image)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            image = upsample(image)
            height, width = image.shape[2:]
            top = (skip.shape[2] - height) // 2
            left = (skip.shape[3] - width) // 2
            skip = skip[:, :, top:top + height, left:left + width]
            image = decoder(torch.cat((image, skip), dim=1))
        return self.output(image)
