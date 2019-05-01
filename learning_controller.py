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

from depth_pred_generator import GlobalNet, RefinementNet, CompleteGenerator, StereoNetGenerator, HierarchicalRefinement

import csv
import datetime
from csv_plotter import CSVPlotter
import image_analysis

#TODO: Add time check

class LearningController:
    def __init__(self, use_Jung = True, use_pix2pix = False, use_global_only=False, train_only_ref=False): #if use_Jung is True, then we load its D and G instead of pix2pix
        print("INIT LEARNING CONTROLLER")
        self.cur_index = 0
        self.use_Jung = use_Jung
        self.global_only = use_global_only
        self.use_pix2pix = use_pix2pix
        self.train_only_ref = train_only_ref

    def prepare(self, data_root, use_gpu, load_models, use_discriminator, load_path_G, load_path_D, seed=9876,
                use_hierarchical_refinement=True):
        # PREPARATION BEFORE LEARNING CAN BEGIN!
        ####################################################
        if use_gpu and not torch.cuda.is_available():
            raise Exception("No compatible GPU available")

        self.use_discriminator = use_discriminator

        cudnn.benchmark = True
        torch.manual_seed(seed)
        if use_gpu:
            torch.cuda.manual_seed(seed)
        net_g_model_out_path_global = "checkpoint/{}/netG_global.pth".format("Test1")
        #For the first tests, only load the generator
        if load_models and os.path.isfile(load_path_G) and os.path.isfile(load_path_D) and self.use_pix2pix:
            print("LOAD")
            self.G = torch.load(load_path_G)
            self.D = torch.load(load_path_D)
            # if not self.use_Jung:
            #     self.D = torch.load(load_path_D)
            # else:
            #     self.D = discriminator_model.get_discriminator_model_Jung(use_gpu=True)
            print("Successfully loaded G and D")
        elif load_models and os.path.isfile(net_g_model_out_path_global):
            print("LOAD GLOBAL ONLY")
            self.G_global = torch.load(net_g_model_out_path_global)
            net_g_model_out_path_refinement = "checkpoint/{}/netG_refinement.pth".format("Test1")
            self.G_ref = torch.load(net_g_model_out_path_refinement)
            self.D = torch.load(load_path_D)

            # if use_hierarchical_refinement:
            #     self.G_ref = HierarchicalRefinement()
            # else:
            #     self.G_ref = RefinementNet(3, 1)
            #
            # if self.use_Jung:
            #     self.D = discriminator_model.get_discriminator_model_Jung(use_gpu=True)
            # else:
            #     self.D = discriminator_model.get_discriminator_model(3 + 1, 8, use_gpu)

        else:
            if self.use_pix2pix:
                self.G = generator_model.get_generator_model(3, 1, 32, True)
            else:
                if use_hierarchical_refinement:
                    # self.G = StereoNetGenerator(3, 1)
                    self.G_global = GlobalNet(3, 1, interpol_rate=4)
                    self.G_ref = HierarchicalRefinement()
                else:
                    # self.G = CompleteGenerator(3, 1)
                    self.G_global = GlobalNet(3, 1, interpol_rate=2)
                    self.G_ref = RefinementNet(3, 1)

            if self.use_Jung:
                self.D = discriminator_model.get_discriminator_model_Jung(use_gpu=True)
            else:
                self.D = discriminator_model.get_discriminator_model(3 + 1, 8, use_gpu) #3 is for the condition (RGB) and
                                                                                    # +3 / +1 is for the output image
        #debugging output just for testing
        if self.use_pix2pix:
            print_network(self.G)
        else:
            print_network(self.G_global)
            print_network(self.G_ref)
        print_network(self.D)

        #data managers using the torch data util methods
        self.trainloader = data_manager.DataManager(data_root, train=True, use_small_patches=False)
        self.testloader = data_manager.DataManager(data_root, train=False, use_small_patches=False)

        self.train_generator = data.DataLoader(self.trainloader, batch_size=8, shuffle=True, num_workers=0)
        self.test_generator = data.DataLoader(self.testloader, batch_size=1, shuffle=False, num_workers=0)

        print("AMOUNT OF IMAGES FOR TRAINING: " + str(len(self.trainloader)))
        print("AMOUNT OF IMAGES FOR TESTING: " + str(len(self.testloader)))

        #the loss functions
        #overall objective should be: arg min max LcGAN(G, D) + lambda*L1_loss(G)
        self.gan_loss = gan_loss.GANLoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.last_val_loss = 0 #this is just initialization

        #need two optimizers as we have different numbers of params
        if self.use_pix2pix:
            self.optimizerG = optim.Adam(self.G.parameters(), lr=0.002, betas=(0.5, 0.999))
        else:
            self.optimizerG_global = optim.Adam(self.G_global.parameters(), lr=0.002, betas=(0.5, 0.999))
            self.optimizerG_ref = optim.Adam(self.G_ref.parameters(), lr=0.002, betas=(0.5, 0.999))
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
            if self.use_pix2pix:
                self.G = self.G.cuda()
            else:
                self.G_global = self.G_global.cuda()
                self.G_ref = self.G_ref.cuda()
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
        self.optimizerG = optim.SGD(self.G.parameters(), lr=0.002)

        #TODO:
        # Change loss so network doesn't predict zeros:
        # supervise to the ground truth value

        l1_sum_train = 0
        l1_count = 0

        for iteration, data in enumerate(self.train_generator, 0):
            a_images, b_images = data
            for im in b_images:
                #TODO: Check here whether we actually want to use these images? Or already check in the generator
                im = im[0]
                avg_grad = image_analysis.get_avg_gradient(im)
                rel_zeros = image_analysis.check_for_zeros(im)
                print("AVERAGE GRADIENT: " + str(avg_grad) + " PERCENT ZEROS: " + str(rel_zeros))
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
                fake_b, fake_b_global = self.G(self.real_a)


                # Clear the gradients
                self.optimizerG.zero_grad()

                # Now for G -> get loss and do a step
                # Get fake data from G and push it through D to get its prediction (real of fake)
                # fake_complete = torch.cat((self.real_a, fake_b), 1)

                # Addition of l1 loss
                fake_b = fake_b * (self.real_b > 0).float() #only supervise non-zero
                loss_g_l1 = self.l1_loss(fake_b, self.real_b)

                fake_b_global = fake_b_global * (self.real_b > 0).float()
                l1_only_global = self.l1_loss(fake_b_global, self.real_b)
                weight_only_global = 1

                # get combined loss for G and do backprop + optimizer step
                loss_g_global_weighted = weight_only_global * l1_only_global

                if self.global_only:
                    loss_g = loss_g_global_weighted
                else:
                    loss_g = loss_g_l1 + loss_g_global_weighted


                loss_g.backward()
                self.optimizerG.step()

                print("STEP " + str(i+1) + "/" + str(len(a_images)) + " G-LOSS: " + str(loss_g.data.item()))

                l1_sum_train += loss_g.data.item()
                l1_count += 1

        print("GOT " + str(l1_count) + " VALUES FOR TRAINING LOSS CALCULATION:")

        l1_train = l1_sum_train / l1_count
        print("---> " + str(l1_sum_train) + "/" + str(l1_count) + " = " + str(l1_train))
        self.cur_l1_train = l1_train

        # with open(self.csv_logfile, 'a', newline='') as csvfile:
        #
        #     spamwriter = csv.writer(csvfile, delimiter=',',
        #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow([l1_train, self.last_val_loss])
        # self.plotter.plot_csv_to_image()


    def learn(self, data_root, use_gpu,  load_models, load_path_G, load_path_D, num_epochs = 10, seed=9876):
        #We need to prepare our variables first. I do this in the prepare() method
        # to keep the code a little cleaner
        # self.prepare(data_root, use_gpu, load_models, load_path_G, load_path_D, seed)

        #Actual training starts here
        # enumerate through the trainloader - thanks torch.utils.data, great job!
        #TODO: Use batches properly!
        # iteration = 0
        if self.use_pix2pix:
            self.optimizerG = optim.SGD(self.G.parameters(), lr=0.004)
        else:
            self.optimizerG_global = optim.SGD(self.G_global.parameters(), lr=0.004)
            self.optimizerG_ref = optim.SGD(self.G_ref.parameters(), lr=0.004)
        self.optimizerD = optim.SGD(self.D.parameters(), lr=0.004)

        g_loss = 0
        g_loss_count = 0
        d_loss = 0
        d_loss_count = 0
        for iteration, data in enumerate(self.train_generator, 0):
            a_images, b_images = data

            # for im in b_images:
            #     #TODO: Check here whether we actually want to use these images? Or already check in the generator
            #     im = im[0]
            #     avg_grad = image_analysis.get_avg_gradient(im)
            #     rel_zeros = image_analysis.check_for_zeros(im)
            #     print("AVERAGE GRADIENT: " + str(avg_grad) + " PERCENT ZEROS: " + str(rel_zeros))

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
                if not self.use_pix2pix:
                    # fake_b, fake_b_global = self.G(self.real_a)
                    fake_b_global = self.G_global(self.real_a)
                    fake_b = self.G_ref(self.real_a, fake_b_global)
                else:
                    fake_b = self.G(self.real_a)

                # Do the same with discriminator as with L1 earlier here
                # -> don't learn zero spots!
                # Addition of l1 loss
                print(fake_b.size())
                fake_b = fake_b * (self.real_b > 0).float()  # only supervise non-zero

                if not self.use_pix2pix:
                    fake_b_global = fake_b_global * (self.real_b > 0).float()

                    l1_only_global = self.l1_loss(fake_b_global, self.real_b)
                    weight_only_global = 0

                fake_b = fake_b.cuda()
                #Clear the gradients
                self.optimizerD.zero_grad()
                if self.use_pix2pix:
                    self.optimizerG.zero_grad()
                else:
                    self.optimizerG_global.zero_grad()
                    self.optimizerG_ref.zero_grad()

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
                if self.global_only:
                    gan_weight = 0
                    l1_weight = 1
                else:
                    gan_weight = 1
                    l1_weight = 10

                loss_g_gan = self.gan_loss(pred_fake, True) * gan_weight

                # Addition of l1 loss
                loss_g_l1 = self.l1_loss(fake_b, self.real_b) * l1_weight #In the paper they recommend lambda = 100!

                if not self.use_pix2pix:
                    loss_global_only = l1_only_global

                # get combined loss for G and do backprop + optimizer step
                if not self.global_only:
                    loss_g = loss_g_gan + loss_g_l1
                else:
                    loss_g = loss_global_only

                # if not self.use_pix2pix:
                #     loss_g += loss_global_only

                if self.global_only:
                    print("LOSS G CONSISTING OF L1: " + str(l1_only_global.data.item()))
                    loss_g.backward()
                    self.optimizerG_global.step()
                elif self.train_only_ref:
                    print("LOSS G CONSISTING OF GAN: " + str(loss_g_gan.data.item()) + " AND L1: " + str(loss_g_l1.data.item()))
                    loss_g.backward()
                    self.optimizerG_ref.step()
                    self.optimizerG_global.step()
                else:
                    print("LOSS G CONSISTING OF L1: " + str(l1_only_global.data.item()))
                    loss_g.backward()
                    self.optimizerG.step()

                g_loss += loss_g.data.item()
                g_loss_count += 1
                d_loss += loss_d.data.item()
                d_loss_count += 1

                print("STEP " + str(iteration) + "/" + str(len(a_images)) + ": D-LOSS: "
                      + str(loss_d.data.item()) +
                      " G-LOSS: " + str(loss_g.data.item()))

        print("GOT " + str(g_loss_count) + " VALUES FOR TRAINING LOSS CALCULATION:")

        if self.global_only:
            overall_g_loss = (g_loss / g_loss_count)
        else:
            overall_g_loss = (g_loss / g_loss_count) / 11 # because 10 L1 + 1 Adv

        overall_d_loss = d_loss / d_loss_count
        print("---> " + str(g_loss) + "/" + str(g_loss_count) + " = " + str(overall_g_loss))
        self.cur_l1_train = overall_g_loss
        self.cur_d_loss = overall_d_loss

        # with open(self.csv_logfile, 'a', newline='') as csvfile:
        #     spamwriter = csv.writer(csvfile, delimiter=',',
        #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow([loss_g.data.item()])


    #TODO CHECK HERE IF VALIDATION TESTING IS CORRECT!
    def test(self, use_gpu, epoch_index, generate_imgs=True):
        psnr_sum = 0 # Peak Signal to Noise Ratio:
                #PSNR = 10 * log_10 (MAX^2_I / MSE), max here is 1
        test_counter = 0 #this is for naming the result images in for our test

        l1_sum = 0
        l1_counter = 0
        #same enumeration as for trainloader, but on test data now
        for iteration, data in enumerate(self.test_generator, 0):
            test_counter += 1
            image_name = "epoch" + str(epoch_index) + "_test_image" + str(test_counter) + ".png"
            image_name_input = "test_image" + str(test_counter) + "_INPUT.png"
            image_name_target = "test_image" + str(test_counter) + "_TARGET.png"

            a_images, b_images = data
            a_img = a_images[0]
            b_img = b_images[0]
            with torch.no_grad():
                real_input, real_output = Variable(a_img), Variable(b_img)
                # real_input, real_output = Variable(train_a, train_b)
                if use_gpu:
                    real_input = real_input.cuda()
                    real_output = real_output.cuda()

            input = real_input.unsqueeze(0) #should stay consistently at a/b or input/output
            target = real_output.unsqueeze(0)

            #Generate a prediction with G and calculate mse and psnr from mse
            if self.use_pix2pix:
                prediction = self.G(input)
            else:
                prediction_global_only = self.G_global(input)
                prediction = self.G_ref(input, prediction_global_only)
            # mse = self.mse_loss(prediction, target)

            # prediction = prediction * (input > 0).float()  # only supervise non-zero
            # loss_g_l1 = self.l1_loss(prediction, self.real_b)
            # print("L1 Test: " + str(loss_g_l1))

            if self.global_only:
                sup_pred = prediction_global_only * (target > 0).float()
            else:
                sup_pred = prediction * (target > 0).float()  # only supervise non-zero

            l1 = self.l1_loss(sup_pred, target)
            l1_loss = l1.data.item()
            l1_sum += l1_loss
            l1_counter += 1
            # psnr = 10 * log10(1 / mse.data.item())
            # psnr_sum += psnr

            # Generate an image, this is not necessary all the time
            generate_imgs = True
            if generate_imgs:
                if test_counter % 1 == 0:
                    if self.global_only:
                        prediction = prediction_global_only.cpu()
                    else:
                        prediction = prediction.cpu()
                    predicted_img = prediction.data[0]
                    if not os.path.exists(os.path.join("result", "depth")):
                        os.makedirs(os.path.join("result", "depth"))
                    img_util.save_tensor_as_image(predicted_img, "result/{}/{}".format("depth", image_name))
                    img_util.save_tensor_as_image(input.data[0], "result/{}/{}".format("depth", image_name_input))
                    img_util.save_tensor_as_image(target.data[0], "result/{}/{}".format("depth", image_name_target))

        # average_psnr = psnr_sum / len(self.testloader)
        # print("Average PSNR = {:.6f}".format(average_psnr))


        print("GOT " + str(l1_counter) + " VALUES FOR VALIDATION LOSS CALCULATION:")
        avg_l1 = l1_sum / l1_counter
        self.last_val_loss = avg_l1

        print("---> " + str(l1_sum) + "/" + str(l1_counter) + " = " + str(avg_l1))

        print("TRAINING L1 LOSS = " + str(self.cur_l1_train))
        print("VALIDATION L1 LOSS = " + str(avg_l1))

        with open(self.csv_logfile, 'a', newline='') as csvfile:

            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if self.use_discriminator:
                spamwriter.writerow([self.cur_l1_train, self.last_val_loss, self.cur_d_loss])
            else:
                spamwriter.writerow([self.cur_l1_train, self.last_val_loss])
        # self.plotter.plot_csv_to_image()

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

        if self.use_pix2pix:
            net_g_model_out_path = "checkpoint/{}/netG_model.pth".format("Test1")
            net_d_model_out_path = "checkpoint/{}/netD_model.pth".format("Test1")
            torch.save(self.G, net_g_model_out_path)
            torch.save(self.D, net_d_model_out_path)
        else:
            net_g_model_out_path_global = "checkpoint/{}/netG_global.pth".format("Test1")
            net_g_model_out_path_refinement = "checkpoint/{}/netG_refinement.pth".format("Test1")
            net_d_model_out_path = "checkpoint/{}/netD_model.pth".format("Test1")
            torch.save(self.G_global, net_g_model_out_path_global)
            torch.save(self.G_ref, net_g_model_out_path_refinement)
            torch.save(self.D, net_d_model_out_path)

        print("Checkpoint saved to {}".format("checkpoint/" + "Test1"))
