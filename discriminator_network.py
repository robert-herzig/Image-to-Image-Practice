import torch.nn as nn
import torch
import numpy as np

#patch-gan == convnet ?
# PATCH GAN:
# The difference between a PatchGAN and regular GAN discriminator is
# that rather the regular GAN maps from a 256x256 image to a single scalar
# output, which signifies "real" or "fake", whereas the PatchGAN maps from
# 256x256 to an NxN array of outputs X, where each X_ij signifies whether
# the patch ij in the image is real or fake.
# quote: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
class get_patch_GAN_discriminator(nn.Module):
    def __init__(self, num_channels, num_filters=64,
                 num_layers=3):
        super(get_patch_GAN_discriminator, self).__init__()

        kernel_width = 4
        padding_width = int(np.ceil((kernel_width-1)/2)) #standard, I suppose

        blocks = [
            nn.Conv2d(num_channels, num_filters,
                kernel_size=kernel_width, stride=2,
                padding=padding_width),
            nn.LeakyReLU(0.2, True)
        ]

        filter_multiplier = 1
        num_filters_mult_prev = 1

        #Not sure whether this is alright style, I think it's more readable this way
        def get_single_block():
            return [nn.Conv2d(num_filters * num_filters_mult_prev, num_filters * filter_multiplier,
                        kernel_size=kernel_width, stride=2,
                        padding=padding_width),
                    nn.BatchNorm2d(num_filters * filter_multiplier, affine=True),
                    nn.LeakyReLU(0.2, True)]

        #get the n layers defined in the argument
        for n in range(1, num_layers):
            num_filters_mult_prev = filter_multiplier
            filter_multiplier = min(2**n, 8)
            blocks += get_single_block() #see function above

        num_filters_mult_prev = filter_multiplier
        filter_multiplier = min(2 ** num_layers, 8)
        blocks += [
            nn.Conv2d(num_filters * num_filters_mult_prev, num_filters * filter_multiplier,
                      kernel_size=kernel_width, stride=1,
                      padding=padding_width),
            nn.BatchNorm2d(num_filters * filter_multiplier, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        blocks += [nn.Conv2d(num_filters * filter_multiplier, 1,
                             kernel_size=kernel_width, stride=1,
                             padding=padding_width)]

        self.model = nn.Sequential(*blocks)

    def forward(self, input):
        return self.model(input)