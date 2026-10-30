import torch
from torch import nn
from torch.nn import functional as F


def spatial_tuple(value):
    """convert a scalar or sequence to a spatial tuple"""
    return (value,) * 3 if isinstance(value, int) else tuple(value) 

class ResidualBlock(nn.Module):
    def __init__(self, channels, settings):
        """build a 3D residual branch with normalization and dropout"""
        super().__init__()
        kernel = spatial_tuple(settings["kernel_size"])
        # preserve spatial size for odd kernels
        padding = tuple(k // 2 for k in kernel) 
        norm = {"eps": settings["norm_epsilon"], "affine": settings["norm_affine"]}
        self.main = nn.Sequential(
            nn.InstanceNorm3d(channels, **norm),
            nn.LeakyReLU(settings["negative_slope"]),
            nn.Conv3d(channels, channels, kernel, padding=padding, bias=settings["bias"]),
            nn.Dropout3d(settings["dropout"]),
            nn.InstanceNorm3d(channels, **norm),
            nn.LeakyReLU(settings["negative_slope"]),
            nn.Conv3d(channels, channels, kernel, padding=padding, bias=settings["bias"]),
        )

    def forward(self, image):
        """add residual features to the input"""
        return self.main(image) + image  # add the residual branch to the unchanged input


class ResidualUNet3D(nn.Module):
    def __init__(self, settings, in_channels, classes):
        """build the residual 3D U-Net encoder, decoder, and output layers"""
        super().__init__()
        channels, strides = settings["channels"], settings["strides"]
        kernel = spatial_tuple(settings["kernel_size"])
        padding = tuple(k // 2 for k in kernel)
        norm = {"eps": settings["norm_epsilon"], "affine": settings["norm_affine"]}
        self.input = nn.Conv3d(in_channels, channels[0], kernel, padding=padding, bias=settings["bias"])  # project input channels to features
        self.encoders = nn.ModuleList([
            ResidualBlock(channel, settings) for channel in channels
        ])
        self.downsamples = nn.ModuleList([  # reduce spatial size while increasing channels
            nn.Conv3d(before, after, kernel, spatial_tuple(stride), padding, bias=settings["bias"])
            for before, after, stride in zip(channels[:-1], channels[1:], strides)
        ])
        self.upsamples, self.decoders = nn.ModuleList(), nn.ModuleList()
        for before, after in zip(reversed(channels[1:]), reversed(channels[:-1])):
            self.upsamples.append(nn.Sequential(
                nn.Conv3d(before, after, kernel, padding=padding, bias=settings["bias"]),
                nn.InstanceNorm3d(after, **norm),
                nn.LeakyReLU(settings["negative_slope"]),
            ))
            self.decoders.append(nn.Sequential(
                nn.Conv3d(2 * after, 2 * after, kernel, padding=padding, bias=settings["bias"]),
                nn.InstanceNorm3d(2 * after, **norm),
                nn.LeakyReLU(settings["negative_slope"]),
                nn.Conv3d(2 * after, after, 1, bias=settings["bias"]),  # fuse concatenated decoder and skip channels
                nn.InstanceNorm3d(after, **norm),
                nn.LeakyReLU(settings["negative_slope"]),
            ))
        self.outputs = nn.ModuleList([
            nn.Conv3d(channel, classes, 1, bias=settings["bias"])
            for channel in reversed(channels[:3])
        ])

    def forward(self, image):
        """encode and decode a volume and combine multiscale class logits"""
        image = self.encoders[0](self.input(image))
        skips = [image]  # retain encoder features for decoder skip connections
        for downsample, encoder in zip(self.downsamples, self.encoders[1:]):
            image = encoder(downsample(image))
            skips.append(image)
        features = []
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            # match skip dimensions even for odd input sizes
            image = F.interpolate(image, size=skip.shape[2:], mode="nearest")  
            # concatenate along the channel axis
            image = decoder(torch.cat((upsample(image), skip), dim=1))  
            features.append(image)
        logits = self.outputs[0](features[1])  
        for output, feature in zip(self.outputs[1:], features[2:]):
            logits = F.interpolate(logits, size=feature.shape[2:], mode="nearest") + output(feature)
        return logits  
