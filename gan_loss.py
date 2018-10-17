import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

#Final objective according to paper is:
# G* = arg min max LcGAN(G, D) + lambda*L1_loss(G)
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

        self.real_label = 1.0
        self.fake_label = 0.0
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = torch.FloatTensor

        #Just assume LSGAN is the better option here
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_real):
        #target is a real sample
        if target_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else: #target is a generated sample
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_real):
        target_tensor = self.get_target_tensor(input, target_real)
        target_tensor = target_tensor.cuda()
        loss_val = self.loss(input, target_tensor)
        return loss_val