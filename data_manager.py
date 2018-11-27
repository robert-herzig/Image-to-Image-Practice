from os import listdir
from os.path import join

import PIL
import torchvision.transforms as transforms

import torch.utils.data as data

#This class will be used to load data from the specified folder and
# retrieve it using the index. Maybe get an elegant method for iterating
# through it without having to load it all at once (see Generators from TF)
class DataManager(data.Dataset):
    def __init__(self, folder_path, train):
        super(DataManager, self).__init__()
        print("Created Data Manager")
        self.folder_path = folder_path

        if train:
            self.path_base = join(self.folder_path, "train")
        else:
            self.path_base = join(self.folder_path, "test")

        self.a_path = join(self.path_base, "a")
        self.b_path = join(self.path_base, "b")
        # self.generate_all_paths()

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        self.generate_all_paths()


    # TODO: This might be a bad solution once folders get too big -> work in batches then or generator?
    # ARE THERE EVEN GENERATORS LIKE IN TF?
    def generate_all_paths(self):
        print("Getting image paths...")

        print("TRAIN A B:")
        self.a_images = self.get_all_imgs_in_folder(self.a_path)
        self.b_images = self.get_all_imgs_in_folder(self.b_path)

    #TODO: Just use a list of file endings maybe for cleaner code
    def get_all_imgs_in_folder(self, folder):
        file_paths = []
        for file in listdir(folder):
            if file.endswith(".jpg") or file.endswith(".jpeg" or file.endswith(".png")
                                     or file.endswith(".JPG") or file.endswith(".JPEG")
                                     or file.endswith(".PNG")):
                    file_paths.append(join(folder, file))
        return file_paths

    def get_imgs_at_index(self, index):
        a = self.transform(PIL.Image.open(self.a_images[index]).resize((256, 256), PIL.Image.BICUBIC))
        b = self.transform(PIL.Image.open(self.b_images[index]).resize((256, 256), PIL.Image.BICUBIC))

        # a = self.transform(PIL.Image.open(self.a_images[index]).resize((240, 320), PIL.Image.BICUBIC))
        # b = self.transform(PIL.Image.open(self.b_images[index]).resize((240, 320), PIL.Image.BICUBIC))
        return a, b

    def __getitem__(self, index):
        a, b = self.get_imgs_at_index(index)
        return a, b

    def __len__(self):
        return len(self.a_images)

