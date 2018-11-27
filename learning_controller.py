import data_manager
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import discriminator_model, generator_model
import gan_loss
from model_utils import print_network
from math import log10

import os
import img_util

from torch.utils import data

class LearningController:
    def __init__(self):
        print("INIT LEARNING CONTROLLER")
        self.cur_index = 0


    def prepare(self, data_root, use_gpu, load_models, load_path_G, load_path_D, seed=9876):
        # PREPARATION BEFORE LEARNING CAN BEGIN!
        ####################################################
        if use_gpu and not torch.cuda.is_available():
            raise Exception("No compatible GPU available")

        cudnn.benchmark = True
        torch.manual_seed(seed)
        if use_gpu:
            torch.cuda.manual_seed(seed)

        if load_models and os.path.isfile(load_path_G) and os.path.isfile(load_path_D):
            print("LOAD")
            self.G = torch.load(load_path_G)
            self.D = torch.load(load_path_D)
            print("Successfully loaded G and D")
        else:
            #Create G and D, save them as class variables for later use (test etc)
            self.G = generator_model.get_generator_model(3, 1, 8, use_gpu) #TODO: This 1 is for depth tests
            self.D = discriminator_model.get_discriminator_model(3 + 1, 8, use_gpu) #3 is for the condition (RGB) and
                                                                                    # +3 / +1 is for the output image
        #debugging output just for testing
        print_network(self.G)
        print_network(self.D)

        #data managers using the torch data util methods
        self.trainloader = data_manager.DataManager(data_root, train=True)
        self.testloader = data_manager.DataManager(data_root, train=False)

        self.train_data_generator = data.DataLoader(self.trainloader, batch_size=32)
        self.test_data_generator = data.DataLoader(self.testloader, batch_size=32)

        #the loss functions
        #overall objective should be: arg min max LcGAN(G, D) + lambda*L1_loss(G)
        self.gan_loss = gan_loss.GANLoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        #need two optimizers as we have different numbers of params
        self.optimizerG = optim.Adam(self.G.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=0.002, betas=(0.5, 0.999))

        #init a and b vars, we need them for our two input images
        self.real_a = torch.FloatTensor(1, 3, 256, 256)
        self.real_b = torch.FloatTensor(1, 3, 256, 256)

        #FOR DEPTH TEST!!!
        # self.real_a = torch.FloatTensor(1, 3, 240, 320)
        # self.real_b = torch.FloatTensor(1, 3, 240, 320)

        #get all the relevant variables ready for the gpu
        if use_gpu:
            self.D = self.D.cuda()
            self.G = self.G.cuda()
            self.gan_loss = self.gan_loss.cuda()
            self.l1_loss = self.l1_loss.cuda()
            self.mse_loss = self.mse_loss.cuda()
            self.real_a = self.real_a.cuda()
            self.real_b = self.real_b.cuda()

        #Transpose the a and b tensors to Variables to make them usable
        self.real_a = Variable(self.real_a)
        self.real_b = Variable(self.real_b)

    def learn(self, data_root, use_gpu,  load_models, load_path_G, load_path_D, num_epochs = 10, seed=9876):
        #We need to prepare our variables first. I do this in the prepare() method
        # to keep the code a little cleaner
        # self.prepare(data_root, use_gpu, load_models, load_path_G, load_path_D, seed)

        #Actual training starts here
        # enumerate through the trainloader - thanks torch.utils.data, great job!
        #TODO: Use batches properly!
        # iteration = 0
        for iteration, batch in enumerate(self.trainloader, 1):
            # for train_a, train_b in self.train_data_generator:
            # real_input_cpu, real_output_cpu = batch[0], batch[1] #retrieve a and b from our loader
            #TODO: Do I have to put torch.no_grad() around much more or only this block?
            with torch.no_grad():
                real_input, real_output = Variable(batch[0]), Variable(batch[1])
                # real_input, real_output = Variable(train_a, train_b)
                if use_gpu:
                    real_input = real_input.cuda()
                    real_output = real_output.cuda()

            self.real_a = real_input.unsqueeze(0) #should stay consistently at a/b or input/output
            self.real_b = real_output.unsqueeze(0)

            #let G generate the fake image
            fake_b = self.G(self.real_a)

            #Clear the gradients
            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()

            #fake_complete is concatenation of real input and the fake (generated) output
            #The Discriminator is also conditioned on real a
            fake_complete = torch.cat((self.real_a, fake_b), 1)
            #pred_fake is then the prediction of the discriminator on the fake output
            pred_fake = self.D.forward(fake_complete.detach())
            #this is the normal gan-loss on D's prediction
            loss_d_fake = self.gan_loss(pred_fake, False)

            #do the same as before but with the real output
            real_complete = torch.cat((self.real_a, self.real_b), 1)
            pred_real = self.D.forward(real_complete)
            loss_d_real = self.gan_loss(pred_real, True)

            #Get loss for D and do a step here
            loss_d = (loss_d_fake + loss_d_real) / 2
            #backprop
            loss_d.backward()
            self.optimizerD.step()

            #Now for G -> get loss and do a step
            #Get fake data from G and push it through D to get its prediction (real of fake)
            fake_complete = torch.cat((self.real_a, fake_b), 1)
            pred_fake = self.D.forward(fake_complete)
            loss_g_gan = self.gan_loss(pred_fake, True)

            # Addition of l1 loss
            loss_g_l1 = self.l1_loss(fake_b, self.real_b) * 2 #In the paper they recommend lambda = 100!

            # get combined loss for G and do backprop + optimizer step
            loss_g = loss_g_gan + loss_g_l1
            print("LOSS G CONSISTING OF GAN: " + str(loss_g_gan.data.item()) + " L1: " + str(loss_g_l1.data.item()))
            loss_g.backward()
            self.optimizerG.step()

            print("STEP " + str(iteration) + "/" + str(len(self.trainloader)) + ": D-LOSS: " + str(loss_d.data.item()) +
                  " G-LOSS: " + str(loss_g.data.item()))

    def test(self, use_gpu, epoch_index, generate_imgs=True):
        psnr_sum = 0 # Peak Signal to Noise Ratio:
                #PSNR = 10 * log_10 (MAX^2_I / MSE), max here is 1
        test_counter = 0 #this is for naming the result images in for our test

        #same enumeration as for trainloader, but on test data now
        for batch in self.testloader:
            test_counter += 1
            image_name = "epoch" + str(epoch_index) + "_test_image" + str(test_counter) + ".jpg"

            #Get real data again, just like in learn()
            with torch.no_grad():
                input, target = Variable(batch[0]), Variable(batch[1])
                if use_gpu:
                    input = input.cuda()
                    target = target.cuda()

            input = input.unsqueeze(0)
            target = target.unsqueeze(0)

            #Generate a prediction with G and calculate mse and psnr from mse
            prediction = self.G(input)
            mse = self.mse_loss(prediction, target)
            psnr = 10 * log10(1 / mse.data.item())
            psnr_sum += psnr

            #if you do not want to generate images, skip the rest of the loop
            if not generate_imgs:
                continue

            # Generate an image, this is not necessary all the time
            if test_counter % 10 == 0:
                prediction = prediction.cpu()
                predicted_img = prediction.data[0]
                if not os.path.exists(os.path.join("result", "depth")):
                    os.makedirs(os.path.join("result", "depth"))
                img_util.save_tensor_as_image(predicted_img, "result/{}/{}".format("depth", image_name))

        average_psnr = psnr_sum / len(self.testloader)
        print("Average PSNR = {:.6f}".format(average_psnr))

    def checkpoint(self, epoch):
        #create directory for checkpoints
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")

        #just for testing, I call this checkpoint folder Test1
        if not os.path.exists(os.path.join("checkpoint", "Test1")):
            os.mkdir(os.path.join("checkpoint", "Test1"))

        #Save G and D into separate pth-files
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format("Test1", epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format("Test1", epoch)
        torch.save(self.G, net_g_model_out_path)
        torch.save(self.D, net_d_model_out_path)

        print("Checkpoint saved to {}".format("checkpoint" + "Test1"))
