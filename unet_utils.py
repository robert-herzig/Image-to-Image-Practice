
import torch
import torch.nn as nn
import torch.nn.functional as F


#Built from:
# Conv2d -> Batch Normalization -> ReLU
class double_conv_block(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(double_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, 3, padding=0),
            nn.BatchNorm2d(num_channels_out),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_out, num_channels_out, 3, padding=0),
            nn.BatchNorm2d(num_channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv_block(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(inconv_block, self).__init__()
        self.conv = double_conv_block(num_channels_in, num_channels_out)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_block(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(down_block, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_block(num_channels_in, num_channels_out)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_block(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, bilinear=True):
        super(up_block, self).__init__()

        if bilinear:
            #Somehow replace this with the nn.functional.interpolate() function
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(num_channels_in // 2, num_channels_in // 2, 2, stride=2)

        self.conv = double_conv_block(num_channels_in, num_channels_out)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv_block(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(outconv_block, self).__init__()
        self.conv = nn.Conv2d(num_channels_in, num_channels_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return x