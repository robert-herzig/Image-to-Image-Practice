import torch.nn as nn
import numpy as np

#patch-gan == convnet ?
# PATCH GAN:
# The difference between a PatchGAN and regular GAN discriminator is
# that rather the regular GAN maps from a 256x256 image to a single scalar
# output, which signifies "real" or "fake", whereas the PatchGAN maps from
# 256x256 to an NxN array of outputs X, where each X_ij signifies whether
# the patch ij in the image is real or fake.
# quote: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39

# -> Faster, training with larger images, results equally good or better than normal GAN

class get_patch_GAN_discriminator(nn.Module):
    def __init__(self, num_channels, num_filters=64,
                 num_layers=3):
        super(get_patch_GAN_discriminator, self).__init__()

        kernel_width = 4
        padding_width = int(np.ceil((kernel_width-1)/2)) #standard, I think

        self.num_filters = num_filters
        self.filter_multiplier = 1
        self.num_filters_mult_prev = 1

        blocks = self.get_start_block(num_channels, kernel_width, padding_width)

        #get the n layers defined in the argument
        for n in range(1, num_layers):
            self.num_filters_mult_prev = self.filter_multiplier
            self.filter_multiplier = min(2**n, 8)
            blocks += self.get_single_block(2, kernel_width, padding_width)

        self.num_filters_mult_prev = self.filter_multiplier
        self.filter_multiplier = min(2 ** num_layers, 8)

        blocks += self.get_single_block(1, kernel_width, padding_width)

        blocks += self.get_final_block(kernel_width, padding_width)

        self.model = nn.Sequential(*blocks)

    def get_start_block(self, num_channels, kernel_width, padding_width):
        return [nn.Conv2d(num_channels, self.num_filters,
                      kernel_size=kernel_width, stride=2,
                      padding=padding_width),
            nn.LeakyReLU(0.2, True)]


    def get_single_block(self, stride, kernel_width, padding_width):
        return [nn.Conv2d(self.num_filters * self.num_filters_mult_prev,
                          self.num_filters * self.filter_multiplier,
                          kernel_size=kernel_width, stride=stride,
                          padding=padding_width),
                nn.BatchNorm2d(self.num_filters * self.filter_multiplier, affine=True),
                nn.LeakyReLU(0.2, True)]

    def get_final_block(self, kernel_width, padding_width):
        return [nn.Conv2d(self.num_filters * self.filter_multiplier, 1,
                   kernel_size=kernel_width, stride=1,
                   padding=padding_width)]

    def forward(self, input):
        return self.model(input)