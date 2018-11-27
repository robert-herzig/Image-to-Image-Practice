import torch.nn as nn
from unet_utils import *


#adapted version from
#https://github.com/milesial/Pytorch-UNet/
#sorry, UNet was really weird for me to implement
#and this code is very good looking
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv_block(n_channels, 64)
        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 512)
        # self.down5 = down_block(512, 512)
        # self.down6 = down_block(512, 512)

        # self.up1 = up_block(1024, 512)
        # self.up2 = up_block(1024, 512)
        self.up3 = up_block(1024, 256)
        self.up4 = up_block(512, 128)
        self.up5 = up_block(256, 64)
        self.up6 = up_block(128, 64)
        self.outc = outconv_block(64, n_classes)

    def forward(self, x):
        #Down
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.down5(x5)
        # x7 = self.down6(x6)
        #
        # #Up
        # x = self.up1(x7, x6)
        # x = self.up2(x, x5)
        x = self.up3(x5, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.outc(x)
        return x