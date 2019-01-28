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

from depth_pred_generator import GlobalNet, RefinementNet, CompleteGenerator

import csv
import datetime
from csv_plotter import CSVPlotter

#TODO: Add time check

class LearningController:
    def __init__(self, use_Jung = True): #if use_Jung is True, then we load its D and G instead of pix2pix
        print("INIT LEARNING CONTROLLER")
        self.cur_index = 0
        self.use_Jung = use_Jung

    def prepare(self, data_root, use_gpu, load_models, use_discriminator, load_path_G, load_path_D, seed=9876):
        # PREPARATION BEFORE LEARNING CAN BEGIN!
        ####################################################
        if use_gpu and not torch.cuda.is_available():
            raise Exception("No compatible GPU available")

        cudnn.benchmark = True
        torch.manual_seed(seed)
        if use_gpu:
            torch.cuda.manual_seed(seed)

        #For the first tests, only load the generator
        if load_models and os.path.isfile(load_path_G) and os.path.isfile(load_path_D):
            print("LOAD")
            self.G = torch.load(load_path_G)
            if not self.use_Jung:
                self.D = torch.load(load_path_D)
            else:
                self.D = discriminator_model.get_discriminator_model_Jung(use_gpu=True)
            print("Successfully loaded G and D")
        else:
            self.G = CompleteGenerator(3, 1)
            if self.use_Jung:
                self.D = discriminator_model.get_discriminator_model_Jung(use_gpu=True)
            else:
                self.D = discriminator_model.get_discriminator_model(3 + 1, 8, use_gpu) #3 is for the condition (RGB) and
                                                                                    # +3 / +1 is for the output image
        #debugging output just for testing
        print_network(self.G)
        print_network(self.D)

        #data managers using the torch data util methods
        self.trainloader = data_manager.DataManager(data_root, train=True, use_small_patches=use_discriminator)
        self.testloader = data_manager.DataManager(data_root, train=False, use_small_patches=use_discriminator)

        self.train_generator = data.DataLoader(self.trainloader, batch_size=1, shuffle=True, num_workers=0)
        self.test_generator = data.DataLoader(self.testloader, batch_size=1, shuffle=True, num_workers=0)

        print("AMOUNT OF IMAGES FOR TRAINING: " + str(len(self.trainloader)))
        print("AMOUNT OF IMAGES FOR TESTING: " + str(len(self.testloader)))

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

        #Logging
        now = datetime.datetime.now()
        timestring = "log_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
        timestring = timestring + "_" + str(now.second)
        print(timestring)
        if not os.path.exists(os.path.join("result", "logs")):
            os.makedirs(os.path.join("result", "logs"))

        self.csv_logfile = os.path.join("result", "logs", timestring + ".csv")
        print(self.csv_logfile)
        self.plotter = CSVPlotter(self.csv_logfile, os.path.join("result", "logs", timestring + ".png"))


    def train_only_global_generator(self, use_gpu):
        self.optimizerG = optim.SGD(self.G.parameters(), lr=0.004, momentum=0.9)

        #TODO:
        # Change loss so network doesn't predict zeros:
        # supervise to the ground truth value


        # for iteration, data in enumerate(self.train_generator, 0):
        #     a, b = data
        #     print(a)
        #     print(b)

        for iteration, data in enumerate(self.train_generator, 0):
            a_images, b_images = data
            print("BATCH " + str(iteration+1) + "/" + str(len(self.train_generator)))
            for i in range(0, len(a_images)):
                with torch.no_grad():
                    real_input, real_output = Variable(a_images[i]), Variable(b_images[i])
                    # real_input, real_output = Variable(train_a, train_b)
                    if use_gpu:
                        real_input = real_input.cuda()
                        real_output = real_output.cuda()

                self.real_a = real_input.unsqueeze(0)  # should stay consistently at a/b or input/output
                self.real_b = real_output.unsqueeze(0)

                # let G generate the fake image
                fake_b = self.G(self.real_a)

                # Clear the gradients
                self.optimizerG.zero_grad()

                # Now for G -> get loss and do a step
                # Get fake data from G and push it through D to get its prediction (real of fake)
                # fake_complete = torch.cat((self.real_a, fake_b), 1)

                # Addition of l1 loss
                fake_b = fake_b * (self.real_b > 0).float() #only supervise non-zero
                loss_g_l1 = self.l1_loss(fake_b, self.real_b)# In the paper they recommend lambda = 100!

                # get combined loss for G and do backprop + optimizer step
                loss_g = loss_g_l1

                loss_g.backward()
                self.optimizerG.step()

                print("STEP " + str(i+1) + "/" + str(len(a_images)) + " G-LOSS: " + str(loss_g.data.item()))

        with open(self.csv_logfile, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([loss_g.data.item()])
        # self.plotter.plot_csv_to_image()


    def learn(self, data_root, use_gpu,  load_models, load_path_G, load_path_D, num_epochs = 10, seed=9876):
        #We need to prepare our variables first. I do this in the prepare() method
        # to keep the code a little cleaner
        # self.prepare(data_root, use_gpu, load_models, load_path_G, load_path_D, seed)

        #Actual training starts here
        # enumerate through the trainloader - thanks torch.utils.data, great job!
        #TODO: Use batches properly!
        # iteration = 0
        self.optimizerG = optim.SGD(self.G.parameters(), lr=0.001)
        self.optimizerD = optim.SGD(self.D.parameters(), lr=0.001)

        for iteration, data in enumerate(self.train_generator, 0):
            a_images, b_images = data
            print("BATCH " + str(iteration+1) + "/" + str(len(self.train_generator)))
            for i in range(0, len(a_images)):

                with torch.no_grad():
                    real_input, real_output = Variable(a_images[i]), Variable(b_images[i])
                    # real_input, real_output = Variable(train_a, train_b)
                    if use_gpu:
                        real_input = real_input.cuda()
                        real_output = real_output.cuda()

                self.real_a = real_input.unsqueeze(0) #should stay consistently at a/b or input/output
                self.real_b = real_output.unsqueeze(0)

                #let G generate the fake image
                fake_b = self.G(self.real_a)

                # Do the same with discriminator as with L1 earlier here
                # -> don't learn zero spots!
                # Addition of l1 loss
                fake_b = fake_b * (self.real_b > 0).float()  # only supervise non-zero

                fake_b = fake_b.cuda()
                #Clear the gradients
                self.optimizerD.zero_grad()
                self.optimizerG.zero_grad()

                if self.use_Jung:
                    print("USE JUNG DISCRIMINATOR")
                    pred_fake = self.D.forward(self.real_a, fake_b)
                    loss_d_fake = self.gan_loss(pred_fake, False)

                    pred_real = self.D.forward(self.real_a, self.real_b)
                    loss_d_real = self.gan_loss(pred_real, True)
                else:
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

                loss_d_fake = loss_d_fake.cuda()
                loss_d_real = loss_d_real.cuda()

                #Get loss for D and do a step here
                loss_d = (loss_d_fake + loss_d_real) / 2

                #backprop
                loss_d.backward(retain_graph=True)
                self.optimizerD.step()

                #Now for G -> get loss and do a step
                #Get fake data from G and push it through D to get its prediction (real of fake)
                loss_g_gan = self.gan_loss(pred_fake, True) * 1

                # Addition of l1 loss
                loss_g_l1 = self.l1_loss(fake_b, self.real_b) * 10 #In the paper they recommend lambda = 100!

                # get combined loss for G and do backprop + optimizer step
                loss_g = loss_g_gan + loss_g_l1
                print("LOSS G CONSISTING OF GAN: " + str(loss_g_gan.data.item()) + " AND L1: " + str(loss_g_l1.data.item()))
                loss_g.backward()
                self.optimizerG.step()

                print("STEP " + str(iteration) + "/" + str(len(self.trainloader)) + ": D-LOSS: "
                      + str(loss_d.data.item()) +
                      " G-LOSS: " + str(loss_g.data.item()))

        with open(self.csv_logfile, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([loss_g.data.item()])


    def test(self, use_gpu, epoch_index, generate_imgs=True):
        psnr_sum = 0 # Peak Signal to Noise Ratio:
                #PSNR = 10 * log_10 (MAX^2_I / MSE), max here is 1
        test_counter = 0 #this is for naming the result images in for our test

        #same enumeration as for trainloader, but on test data now
        for batch in self.testloader:
            test_counter += 1
            image_name = "epoch" + str(epoch_index) + "_test_image" + str(test_counter) + ".png"
            image_name_input = "test_image" + str(test_counter) + "_INPUT.png"
            image_name_target = "test_image" + str(test_counter) + "_TARGET.png"

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

            # prediction = prediction * (input > 0).float()  # only supervise non-zero
            # loss_g_l1 = self.l1_loss(prediction, self.real_b)
            # print("L1 Test: " + str(loss_g_l1))

            psnr = 10 * log10(1 / mse.data.item())
            psnr_sum += psnr

            #if you do not want to generate images, skip the rest of the loop
            if not generate_imgs:
                continue

            # Generate an image, this is not necessary all the time
            if test_counter % 1 == 0:
                prediction = prediction.cpu()
                predicted_img = prediction.data[0]
                if not os.path.exists(os.path.join("result", "depth")):
                    os.makedirs(os.path.join("result", "depth"))
                img_util.save_tensor_as_image(predicted_img, "result/{}/{}".format("depth", image_name))
                img_util.save_tensor_as_image(input.data[0], "result/{}/{}".format("depth", image_name_input))
                img_util.save_tensor_as_image(target.data[0], "result/{}/{}".format("depth", image_name_target))

        average_psnr = psnr_sum / len(self.testloader)
        print("Average PSNR = {:.6f}".format(average_psnr))
        self.plotter.plot_csv_to_image()

    def checkpoint(self, epoch):
        #create directory for checkpoints
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")

        #just for testing, I call this checkpoint folder Test1
        if not os.path.exists(os.path.join("checkpoint", "Test1")):
            os.mkdir(os.path.join("checkpoint", "Test1"))

        #Save G and D into separate pth-files
        # net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format("Test1", epoch)
        # net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format("Test1", epoch)


        net_g_model_out_path = "checkpoint/{}/netG_model.pth".format("Test1")
        net_d_model_out_path = "checkpoint/{}/netD_model.pth".format("Test1")
        torch.save(self.G, net_g_model_out_path)
        torch.save(self.D, net_d_model_out_path)

        print("Checkpoint saved to {}".format("checkpoint/" + "Test1"))
