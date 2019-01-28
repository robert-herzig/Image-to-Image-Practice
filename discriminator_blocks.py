import torch
import torch.nn as nn
import torch.nn.functional as F

"""
For RGB, we use 3 input channels, for depth 1
"""
class StartBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(StartBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, 3, stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x

"""
This uses a stride of 2 instead of pooling
"""
class SingleBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(SingleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, 4, stride=(2, 2), padding=0),
            nn.BatchNorm2d(num_channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class FCBlock(nn.Module):
    def __init__(self):
        super(FCBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(16384, 1024), # Does this really have to be this big?!
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.block(x)
        return x

class OutputBlock(nn.Module):
    def __init__(self):
        super(OutputBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.block(x)
        return x

