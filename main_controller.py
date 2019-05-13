import learning_controller

class Main:
    def __init__(self, pix2pix=False, global_only=False, train_only_refinement=False):
        print("Created Main Controller")
        self.learning_controller = learning_controller.LearningController(use_Jung=False, use_pix2pix=pix2pix,
                                                                          use_global_only=global_only,
                                                                          train_only_ref=train_only_refinement)

    def test(self, data_root, num_epochs, load_models, load_path_G, load_path_D):
        self.learning_controller.prepare(data_root, use_gpu=True, load_models=load_models, use_discriminator=True,
                                    load_path_G=load_path_G, load_path_D=load_path_D, use_hierarchical_refinement=True)
        for epoch in range(0, num_epochs):
            print("EPOCH " + str(epoch) + " " + "#"*20)
            self.learning_controller.learn(data_root, use_gpu=True, load_models=load_models, load_path_G=load_path_G, load_path_D=load_path_D)

            if epoch % 10 == 0:
                self.learning_controller.checkpoint(epoch)
            #
            if epoch % 5 == 0:
                self.learning_controller.test(True, epoch)
                self.learning_controller.plotter.plot_csv_to_image(plot_discriminator=True)

    def train_only_global_generator(self, data_root, num_epochs, load_models, load_path_G, load_path_D):
        self.learning_controller.prepare(data_root, use_gpu=True, load_models=load_models, use_discriminator=False,
                                         load_path_G=load_path_G, load_path_D=load_path_D) #TODO: remove D later

        for epoch in range(0, num_epochs):
            print("EPOCH " + str(epoch) + " " + "#"*20)

            self.learning_controller.train_only_global_generator(use_gpu=True)

            if epoch % 5 == 0:
                self.learning_controller.test(True, epoch)
                self.learning_controller.plotter.plot_csv_to_image(plot_discriminator=False)

            if epoch % 50 == 0:
                self.learning_controller.checkpoint(epoch)



if __name__ == '__main__':
    main_controller = Main(pix2pix=False, global_only=False, train_only_refinement=True)
    main_controller.test("datasets/KITTI", 351, False, "checkpoint/Test1/netG_model.pth", "checkpoint/Test1/netD_model.pth")
    # main_controller.train_only_global_generator("datasets/NYU2", 101, False, "checkpoint/Test1/netG_model.pth", "checkpoint/Test1/netD_model.pth")
