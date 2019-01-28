import torch
import model_utils
import discriminator_network

def get_discriminator_model(num_channels_in, num_filters, use_gpu):
    if use_gpu:
        assert(torch.cuda.is_available())

    D = discriminator_network.Discriminator_pix2pix(num_channels_in, num_filters, num_layers=3)

    if use_gpu:
        D.cuda()
    D.apply(model_utils.weights_init)

    return D

def get_discriminator_model_Jung(use_gpu):
    if use_gpu:
        assert(torch.cuda.is_available())

    D = discriminator_network.Discriminator_Jung()

    if use_gpu:
        D.cuda()
    D.apply(model_utils.weights_init)

    return D