import torch
import model_utils
import discriminator_network

def get_discriminator_model(num_channels_in, num_filters, use_gpu):
    if use_gpu:
        assert(torch.cuda.is_available())

    D = discriminator_network.get_patch_GAN_discriminator(num_channels_in, num_filters, num_layers=3)

    if use_gpu:
        D.cuda()
    D.apply(model_utils.weights_init)

    return D