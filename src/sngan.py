import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .conditional_batchnorm import CategoricalConditionalBatchNorm2d

class GeneratorResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
            num_classes=0, upsample=False):
        super(GeneratorResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.upsample = upsample
        self.c1 = nn.Conv2d(in_channels, out_channels,
                kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels,
                kernel_size=3, stride=1, padding=1)
        if self.num_classes > 0:
            self.bn1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_channels)
            self.bn2 = CategoricalConditionalBatchNorm2d(
                num_classes, out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, y=None):
        # compute residual
        if y is not None:
            h = self.bn1(x, y)
        else:
            h = self.bn1(x)
        h = F.relu(h, inplace=True)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.c1(h)
        if y is not None:
            h = self.bn2(h, y)
        else:
            h = self.bn2(h)
        h = F.relu(h, inplace=True)
        h = self.c2(h)

        # compute skip connection
        if self.upsample:
            skip = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.in_channels != self.out_channels:
            skip = self.c_sc(skip)

        return h + skip

class DiscriminatorResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(DiscriminatorResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        self.c1 = spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        self.c2 = spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))

        if self.in_channels != self.out_channels:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # compute residual
        h = F.relu(x)
        h = self.c1(h)
        h = F.relu(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2, stride=2)

        # compute skip connection
        if self.downsample:
            skip = F.avg_pool2d(x, 2, stride=2)
        if self.in_channels != self.out_channels:
            skip = self.c_sc(skip)

        return h + skip

class SNGANProjectionDiscriminator(nn.Module):
    
    def __init__(self, num_classes=0, ndf=128):
        super(SNGANProjectionDiscriminator, self).__init__()

        self.block1 = DiscriminatorResBlock(4, ndf, downsample=True)
        self.block2 = DiscriminatorResBlock(ndf, 2 * ndf, downsample=True)
        self.block3 = DiscriminatorResBlock(2 * ndf, 4 * ndf, downsample=True)
        self.block4 = DiscriminatorResBlock(4 * ndf, 4 * ndf, downsample=True)
        self.linear = nn.Linear(4 * ndf, 1)

        self.l_y = spectral_norm(nn.Embedding(num_classes, 4 * ndf))

    def forward(self, x, y):
        h = x
        for i in range(1, 5):
            h = getattr(self, 'block{}'.format(i))(h)
        h = F.relu(h)
        h = torch.sum(h, dim=(2, 3))
        output = self.linear(h)
        output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)

        return output

class SNGANGenerator(nn.Module):
    
    def __init__(self, num_classes=0,
            input_dim=100, ngf=128,
            bottom_height=4, bottom_width=4):
        super(SNGANGenerator, self).__init__()

        self.num_classes = num_classes
        self.ngf = ngf
        self.bottom_height = bottom_height
        self.bottom_width = bottom_width
        self.preprocess = nn.Linear(
                input_dim, 4 * ngf * bottom_height * bottom_width, bias=False)
        self.block1 = GeneratorResBlock(4 * ngf, 4 * ngf,
                num_classes=self.num_classes, upsample=True)
        self.block2 = GeneratorResBlock(4 * ngf, 2 * ngf,
                num_classes=self.num_classes, upsample=True)
        self.block3 = GeneratorResBlock(2 * ngf, ngf,
                num_classes=self.num_classes, upsample=True)
        self.bn = nn.BatchNorm2d(ngf)
        self.conv4 = nn.Conv2d(ngf, 4, 3, stride=1, padding=1)
        
    def forward(self, z, y):
        h = self.preprocess(z).view(
                z.size(0), -1, self.bottom_height, self.bottom_width)
        for i in range(1, 4):
            h = getattr(self, 'block{}'.format(i))(h, y)
        h = F.relu(self.bn(h))
        return torch.tanh(self.conv4(h))
