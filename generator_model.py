import torch
import model_utils
import unet_generator

def get_generator_model(num_channels_in, num_channels_out, num_filters, use_gpu):
    if use_gpu:
        assert(torch.cuda.is_available())

    G = unet_generator.UNet(num_channels_in, num_channels_out)

    if use_gpu > 0:
        G.cuda()
    G.apply(model_utils.weights_init)

    return G

