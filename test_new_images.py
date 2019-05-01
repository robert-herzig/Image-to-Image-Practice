import torch
from torch.autograd import Variable

from model_utils import print_network
from torch.utils import data

import os
import img_util
import test_only_data_manager

class NetTester:
    def __init__(self, path_global_net, path_refinement_net, path_images, path_output): #if use_Jung is True, then we load its D and G instead of pix2pix
        print("INIT LEARNING CONTROLLER")
        self.index = 0
        self.path_global = path_global_net
        self.path_ref = path_refinement_net
        self.path_images = path_images
        self.path_output = path_output
        self.load_stuff()

    def load_stuff(self):
        print("START LOADING NETS")
        self.G_global = torch.load(self.path_global)
        self.G_ref = torch.load(self.path_ref)

        self.G_global = self.G_global.cuda()
        self.G_ref = self.G_ref.cuda()
        print_network(self.G_global)
        print_network(self.G_ref)

        print("LOADED NETS COMPLETELY")

        print("LOAD IMAGES")
        self.image_loader = test_only_data_manager.DataManager(self.path_images)
        self.data_generator = data.DataLoader(self.image_loader, batch_size=1, shuffle=False, num_workers=0)


    #TODO CHECK HERE IF VALIDATION TESTING IS CORRECT!
    def test(self):
        test_counter = 0
        for iteration, data in enumerate(self.data_generator, 0):
            test_counter += 1

            image_name = "test_image" + str(test_counter) + ".png"
            image_name_input = "test_image" + str(test_counter) + "_INPUT.png"

            a_images = data
            a_img = a_images[0]

            with torch.no_grad():
                real_input = Variable(a_img)
                real_input = real_input.cuda()

            input = real_input.unsqueeze(0) #should stay consistently at a/b or input/output

            prediction_global_only = self.G_global(input)
            prediction = self.G_ref(input, prediction_global_only)

            prediction = prediction.cpu()
            predicted_img = prediction.data[0]

            #TODO output path
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)

            img_util.save_tensor_as_image(predicted_img, os.path.join(self.path_output, image_name))
            img_util.save_tensor_as_image(input.data[0], os.path.join(self.path_output, image_name_input))


if __name__ == '__main__':
    model_path = "C:\\Users\\Rob\\Desktop\\MASSH\\models"
    path_global = model_path + "\\netG_global.pth"
    path_ref = model_path + "\\netG_refinement.pth"
    path_images_in = "C:\\Users\\Rob\\Desktop\\MASSH\\TestingNewImages\\InputImages"
    path_images_out = "C:\\Users\\Rob\\Desktop\\MASSH\\TestingNewImages\\OutputImages"

    tester = NetTester(path_global, path_ref, path_images_in, path_images_out)
    print("START TEST")
    tester.test()
    print("FINISHED TEST")
