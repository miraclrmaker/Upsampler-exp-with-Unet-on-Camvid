""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .carafe import CARAFE_upsample
from .dysample import DySample

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, style1):
        super().__init__()
        self.style1 = style1
        if style1 == 'max-pooling':
            self.down_op = nn.MaxPool2d(2)
        elif style1 == 'max-pooling-indices':
            # MaxPool with indices for MaxUnpooling
            self.down_op = nn.MaxPool2d(2, return_indices=True)
        elif style1 == 'avg-pooling':
            self.down_op = nn.AvgPool2d(2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        if self.style1 == 'max-pooling-indices':
            x_pooled, indices = self.down_op(x)
            x_conv = self.conv(x_pooled)
            return x_conv, indices
        else:
            x_pooled = self.down_op(x)
            x_conv = self.conv(x_pooled)
            return x_conv


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, style2='bilinear'):
        super().__init__()
        self.style2 = style2

        # if bilinear, use the normal convolutions to reduce the number of channels
        if style2 != 'deconvolution':
            if style2 == 'bilinear':
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            elif style2 == 'nearest':
                # Nearest neighbor upsampling
                self.up = nn.Upsample(scale_factor=2, mode='nearest')
            elif style2 == 'maxunpooling':
                # MaxUnpooling requires indices from corresponding MaxPool
                self.up = nn.MaxUnpool2d(2)
            elif style2 == 'carafe':
                self.up = CARAFE_upsample(in_channels//2, in_channels//4)
            elif style2 == 'dysample_lp':
                # DySample with Linear+PixelShuffle style, no dynamic scope
                self.up = DySample(in_channels//2, scale=2, style='lp', groups=4, dyscope=False)
            elif style2 == 'dysample_lp-dynamic':
                # DySample with Linear+PixelShuffle style, with dynamic scope
                self.up = DySample(in_channels//2, scale=2, style='lp', groups=4, dyscope=True)
            elif style2 == 'dysample_pl':
                # DySample with PixelShuffle+Linear style, no dynamic scope
                self.up = DySample(in_channels//2, scale=2, style='pl', groups=4, dyscope=False)
            elif style2 == 'dysample_pl-dynamic':
                # DySample with PixelShuffle+Linear style, with dynamic scope
                self.up = DySample(in_channels//2, scale=2, style='pl', groups=4, dyscope=True)
            # elif style2 == 'guided-upsample':



            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, indices=None):
        if self.style2 == 'maxunpooling':
            # MaxUnpooling requires indices from the corresponding MaxPool
            if indices is None:
                raise ValueError("MaxUnpooling requires indices from the corresponding MaxPool layer")
            x1 = self.up(x1, indices)
        else:
            x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
