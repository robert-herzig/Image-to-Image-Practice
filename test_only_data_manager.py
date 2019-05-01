from os import listdir
from os.path import join

import torchvision.transforms as transforms
import torch.utils.data as data
import cv2

# This class presents a dataset implementation that is used to
# iterate through all data in the folder for new images from
# outside the training set (selfmade photographs etc)
class DataManager(data.Dataset):
    def __init__(self, folder_path):
        super(DataManager, self).__init__()
        print("Created Data Manager")
        self.folder_path = folder_path

        transform_list_a = [transforms.ToTensor()]
        self.transform_a = transforms.Compose(transform_list_a)

        self.generate_all_paths()

    # TODO: This might be a bad solution once folders get too big -> work in batches then or generator?
    # ARE THERE EVEN GENERATORS LIKE IN TF?
    def generate_all_paths(self):
        print("Getting image paths...")
        self.a_images = self.get_all_imgs_in_folder(self.folder_path)


    # TODO: Just use a list of file endings maybe for cleaner code
    def get_all_imgs_in_folder(self, folder):
        file_paths = []
        for file in listdir(folder):
            if file.__contains__(".png"):
                file_paths.append(join(folder, file))
        print(file_paths)
        return file_paths

    def get_img_at_index(self, index):

        a_img = cv2.imread(self.a_images[index])
        a_img = cv2.resize(a_img, (0, 0), fx=0.75, fy=0.75)

        a_img = a_img[30:286, 30:286]
        a_img = a_img[..., ::-1]  # convert BGR (cv2) to RGB (what we want later)
        a_img = a_img.copy()  # remove negative strides

        a = self.transform_a(a_img)
        return a

    def __getitem__(self, index):
        a = self.get_img_at_index(index)
        return a

    def __len__(self):
        return len(self.a_images)

