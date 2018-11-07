import learning_controller

class Main:
    def __init__(self):
        print("Created Main Controller")
        self.learning_controller = learning_controller.LearningController()

    def test(self, data_root, num_epochs, load_models, load_path_G, load_path_D):
        for epoch in range(0, num_epochs):
            print("EPOCH " + str(epoch) + " " + "#"*20)
            self.learning_controller.learn(data_root, use_gpu=True, load_models=load_models, load_path_G=load_path_G, load_path_D=load_path_D)

            if epoch % 10 == 0:
                self.learning_controller.checkpoint(epoch)
        self.learning_controller.test(True)

if __name__ == '__main__':
    main_controller = Main()
    main_controller.test("data/facades", 30, True, "checkpoint/Test1/netG_model_epoch_70.pth", "checkpoint/Test1/netD_model_epoch_70.pth")
