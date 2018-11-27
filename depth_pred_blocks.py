import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Number of channels always gets doubled, according to the paper.
"""
class EncoderBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, 3, padding=0), #it doesn't specify this, right?
            nn.ReLU(inplace=True), #not really sure about inplace
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(DecoderBlock, self).__init__()

        self.up = nn.ConvTranspose2d(num_channels_in, num_channels_in // 2, 2, stride=2)

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in//2, num_channels_out, 3, padding=0), #it doesn't specify this, right?
            nn.ReLU(inplace=True), #not really sure about inplace
        )

    def forward(self, x1):
        x1 = self.up(x1)
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #                 diffY // 2, int(diffY / 2)))
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        return x

class UpsamplingBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(UpsamplingBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.up(x)
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #                 diffY // 2, int(diffY / 2)))
        # x = torch.cat([x2, x1], dim=1)
        # x = self.conv(x)
        return x

class OutconvBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(OutconvBlock, self).__init__()
        self.conv = nn.Conv2d(num_channels_in, num_channels_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
