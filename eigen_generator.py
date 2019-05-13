import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_pred_blocks import OutconvBlock

class EigenGenerator(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(EigenGenerator, self).__init__()

        self.glob = CoarsePredNet(num_channels_in, num_channels_out)
        self.ref = RefinementNet(num_channels_in, num_channels_out)

    def forward(self, x):
        x_glob = self.glob(x)
        x_refined = self.ref(x, x_glob)

        return x_refined, x_glob

"""
Net used for the coarse prediction.
"""
class CoarsePredNet(nn.Module):
    def __init__(self):
        super(CoarsePredNet, self).__init__()
        print("COARSE PRED NET - EIGEN")
        self.net = nn.Sequential(
            ConvBlock(3, 96, 11, stride=4),
            nn.MaxPool2d(2),
            ConvBlock(96, 256, 5),
            nn.MaxPool2d(2),
            ConvBlock(256, 384, 3),
            ConvBlock(384, 384, 3),
            ConvBlock(384, 256, 3),
            ConvBlock(256, 4096, 1),
            ConvBlock(4096, 1, 1),
        )


    def forward(self, x):
        output = self.net(x)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

        return output

"""
Refinement after coarse prediction, uses one layer for feature extraction into a 63 channel matrix 
and concatenates that with the coarse output of the first net.
"""
class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()
        print("REFINEMENT NET - EIGEN")

        #before concat
        self.net1 = nn.Sequential(
            ConvBlock(3, 63, 9, stride=2),
            nn.MaxPool2d(2),
        )

        #after concat
        self.net2 = nn.Sequential(
            ConvBlock(64, 64, 5),
            OutconvBlock(64, 1),
        )

    def forward(self, x_rgb, x_global):
        feature_x = self.net1(x_rgb)

        output = torch.cat([feature_x, x_rgb], dim=1)
        output = self.net2(output)

        return output

"""
Building block for convolutions in this architecture
"""
class ConvBlock(nn.Module):
    def __init__(self, kernel_size, num_channels_in, num_channels_out, stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, kernel_size, padding=0, stride=stride),  # it doesn't specify this, right?
            nn.ReLU(inplace=False),  # not really sure about inplace
        )

    def forward(self, x):
        x = self.block(x)
        return x
