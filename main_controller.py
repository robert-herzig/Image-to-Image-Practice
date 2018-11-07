import learning_controller

class Main:
    def __init__(self):
        print("Created Main Controller")
        self.learning_controller = learning_controller.LearningController()

    def test(self, data_root, num_epochs):
        for epoch in range(0, num_epochs):
            print("EPOCH " + str(epoch) + " " + "#"*20)
            self.learning_controller.learn(data_root, use_gpu=True)

            if epoch % 10 == 0:
                self.learning_controller.checkpoint(epoch)
        self.learning_controller.test(True)

if __name__ == '__main__':
    main_controller = Main()
    main_controller.test("data/facades", 100)
