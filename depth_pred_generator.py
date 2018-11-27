import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_pred_blocks import EncoderBlock, DecoderBlock, UpsamplingBlock, OutconvBlock

"""
For the encoder,
we gradually reduce the spatial resolution with 2 × 2 max-
pooling (stride 2) while doubling the number of channels. The
decoder part predicts the initial depth estimate through a se-
quence of deconvolutional (a factor of 2) and convolutional
layers. The output resolution of the global net is half the in-
put image.
"""
class GlobalNet(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(GlobalNet, self).__init__()

        self.down1 = EncoderBlock(num_channels_in, 64)
        self.down2 = EncoderBlock(64, 128)
        self.down3 = EncoderBlock(128, 256)
        self.down4 = EncoderBlock(256, 512)
        self.down5 = EncoderBlock(512, 1024)
        self.down6 = EncoderBlock(1024, 1024)

        self.up1 = DecoderBlock(1024, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)
        self.up5 = DecoderBlock(64, 3)
        self.upsample = UpsamplingBlock(3, 3)


        self.out = OutconvBlock(3, num_channels_out)

        # self.inc = inconv_block(n_channels, 64)
        # self.down1 = down_block(64, 128)
        # self.down2 = down_block(128, 256)
        # self.down3 = down_block(256, 512)
        # self.down4 = down_block(512, 512)
        # # self.down5 = down_block(512, 512)
        # # self.down6 = down_block(512, 512)
        #
        # # self.up1 = up_block(1024, 512)
        # # self.up2 = up_block(1024, 512)
        # self.up3 = up_block(1024, 256)
        # self.up4 = up_block(512, 128)
        # self.up5 = up_block(256, 64)
        # self.up6 = up_block(128, 64)
        # self.outc = outconv_block(64, n_classes)

    def forward(self, x):
        # Down
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x6 = self.up1(x5)
        x7 = self.up2(x6)
        x8 = self.up3(x7)
        x9 = self.up4(x8)
        x10 = self.up5(x9)
        x11 = self.upsample(x10)

        output = self.out(x11)

        # print("TENSOR SIZES: ")
        # print(x.size())
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        # print(x5.size())
        # print(x6.size())
        # print(x7.size())
        # print(x8.size())
        # print(x9.size())
        # print(x10.size())
        # print(x11.size())
        # print(output.size())


        return output

"""
The refinement net maps the predictions of global net
to the full resolution. It takes the input as the bilinearly-
upsampled (×2) output of the global net. To provide struc-
tural guidance into the refinement net, a feature map 2 from
RGB input is concatenated with the third convolutional layer
of refinement net
"""
class RefinementNet:
    def __init__(self):
        print("REFINEMENT NET IS ALSO GOOD")