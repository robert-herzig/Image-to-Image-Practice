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
            nn.ReLU(inplace=False), #not really sure about inplace
            # nn.ReflectionPad2d(1),
            nn.MaxPool2d(2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.block(x)
        return x


#TODO: For output use this block without Relu at the end
#TODO: Compare using or not using batch normalization
class DecoderBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            # nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_channels_in, num_channels_in, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_channels_in),
            nn.ReLU(inplace=True), #Compare LeakyRelu and Relu - Read LeakyRelu
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.Conv2d(num_channels_in, num_channels_out, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        x = self.block(x1)
        return x

class FinalDecoderBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(FinalDecoderBlock, self).__init__()

        self.block = nn.Sequential(
            # nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_channels_in, num_channels_in, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_channels_in),
            nn.ReLU(inplace=True), #Compare LeakyRelu and Relu - Read LeakyRelu
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.Conv2d(num_channels_in, num_channels_out, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_channels_out),
        )

    def forward(self, x1):
        x = self.block(x1)
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
        return x1

#TODO: Use this as output!!!
class OutconvBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(OutconvBlock, self).__init__()
        self.conv = nn.Conv2d(num_channels_in, num_channels_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class RefinementBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(RefinementBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, 3, padding=0),  # it doesn't specify this, right?
            nn.ReLU(inplace=False),  # not really sure about inplace
            # nn.ReflectionPad2d(1),
            nn.MaxPool2d(2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResidualRefinementBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(ResidualRefinementBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, 3, padding=0),  # it doesn't specify this, right?
            nn.ReLU(inplace=True),  # not really sure about inplace
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(num_channels_out, num_channels_out, 3, padding=0),  # it doesn't specify this, right?
            # nn.ReLU(inplace=False),  # not really sure about inplace

        )


    def forward(self, x1, x2):
        x_full = torch.cat([x1, x2], dim=1)
        # print(x_full.size())
        output = self.block(x_full)
        # print(x_full.size())

        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #                 diffY // 2, int(diffY / 2)))
        # x = torch.cat([x2, x1], dim=1)
        # x = self.conv(x)
        return output

"""
Three convolutional and RELU layers given RGB input
"""
class FeatureBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(FeatureBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, 3, padding=0),  # it doesn't specify this, right?
            nn.ReLU(inplace=False),  # not really sure about inplace
            nn.MaxPool2d(2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LatentVectorBlock(nn.Module):
    def __init__(self):
        super(LatentVectorBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(1 * 512 * 8 * 8, 1 * 512 * 8 * 8)
        )

    def forward(self, x):
        orig_size = x.size
        print("SHAPE BEFORE LATENT BLOCK: " + str(x.size(0)) + "|" + str(x.size(1)) + "|" + str(x.size(2)) +
              "|" + str(x.size(3)))

        x = self.block(x)
        x.view(orig_size(0), orig_size(1), orig_size(2), orig_size(3))
        print("SHAPE AFTER LATENT BLOCK: " + str(x.size(0)) + "|" + str(x.size(1)) + "|" + str(x.size(2)) +
              "|" + str(x.size(3)))
        return x
