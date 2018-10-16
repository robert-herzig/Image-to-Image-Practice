from __future__ import print_function
import os

import torch
from torch.autograd import Variable
import data_manager
import img_util

cuda = True

model = "checkpoint/Test1/netG_model_epoch_2.pth"
netG = torch.load(model)

image_dir = "data\\facades"
image_filenames = [x for x in os.listdir(image_dir)]

testloader = data_manager.DataManager(image_dir, train=False)

count = 0

for batch in testloader:
    image_name = str(count) + ".jpg"
    count += 1
    input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)

    input = input.unsqueeze(0)
    if cuda:
        netG = netG.cuda()
        input = input.cuda()

    out = netG(input)
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists(os.path.join("result", "facades")):
        os.makedirs(os.path.join("result", "facades"))
    img_util.save_tensor_as_image(out_img, "result/{}/{}".format("facades", image_name))
