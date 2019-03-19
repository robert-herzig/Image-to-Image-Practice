import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_pred_blocks import EncoderBlock, DecoderBlock, UpsamplingBlock, OutconvBlock, FeatureBlock, \
    ResidualRefinementBlock, RefinementBlock, FinalDecoderBlock, LatentVectorBlock
from torchvision import models

class CompleteGenerator(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(CompleteGenerator, self).__init__()

        self.glob = GlobalNet(num_channels_in, num_channels_out)
        self.ref = RefinementNet(num_channels_in, num_channels_out)

    def forward(self, x):
        x_glob = self.glob(x)
        x_refined = self.ref(x, x_glob)

        return x_refined, x_glob


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

        self.vgg16 = models.vgg16()
        # self.vgg16.load_state_dict(torch.load("vgg16-397923af.pth"))

        self.encoder = nn.Sequential(
            *list(self.vgg16.features.children())[:-2]
        )

        self.encoder = nn.Sequential(
            self.encoder,
            EncoderBlock(512, 512),
        )

        #try fc layer here ~ around 1000s - 10000s
        # -> latent vector
        # self.fc_segment = nn.Sequential(
        #     LatentVectorBlock()
        # )

        self.decoder = nn.Sequential(
            # DecoderBlock(1024, 512),
            DecoderBlock(512, 512),
            nn.Dropout(0.4),
            DecoderBlock(512, 256),
            nn.Dropout(0.4),
            DecoderBlock(256, 128),
            nn.Dropout(0.4),
            FinalDecoderBlock(128, num_channels_out),
            # DecoderBlock(64, num_channels_out)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        # print("AFTER ENCODER:")
        # print(summary(self.encoder, (3, 256, 256)))
        #
        # print("AFTER DECODER:")
        # print(summary(self.decoder, (512, 8, 8)))
        # x1 = self.fc_segment(x1)
        output = self.decoder(x1)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

        return output

"""
The refinement net maps the predictions of global net
to the full resolution. It takes the input as the bilinearly-
upsampled (×2) output of the global net. To provide struc-
tural guidance into the refinement net, a feature map 2 from
RGB input is concatenated with the third convolutional layer
of refinement net
"""
class RefinementNet(nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super(RefinementNet, self).__init__()
        print("REFINEMENT NET IS ALSO GOOD")

        self.feature_map_net = nn.Sequential(
            FeatureBlock(num_channels_in, 64),
            FeatureBlock(64, 64),
            FeatureBlock(64, 64),
        )

        #This can already be grayscale
        self.refinement_net1 = nn.Sequential(
            RefinementBlock(num_channels_out, 64),
            nn.Dropout(0.2),
            RefinementBlock(64, 64),
            nn.Dropout(0.2),
            RefinementBlock(64, 64),
            nn.Dropout(0.2),
        )

        self.res_refinement = ResidualRefinementBlock(64+64, 64)

        self.refinement_net3 = nn.Sequential(
            DecoderBlock(64, 64),
            nn.Dropout(0.2),
            DecoderBlock(64, 64),
            FinalDecoderBlock(64, num_channels_out)
            # DecoderBlock(64, num_channels_out),
            # OutconvBlock(num_channels_out, num_channels_out)
        )

    def forward(self, x_rgb, x_global):
        # print(x.size())
        feature_x = self.feature_map_net(x_rgb)
        refinement1_x = self.refinement_net1(x_global)

        residual_refinement = self.res_refinement(refinement1_x, feature_x)

        # x_global = self.feature_map_net(x_global) #this is just for testing
        refinement2_x = self.refinement_net3(residual_refinement)

        return refinement2_x
