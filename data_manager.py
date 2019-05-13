from os import listdir
from os.path import join

import PIL
import torchvision.transforms as transforms

import torch.utils.data as data

import torch
import cv2
import numpy as np

import random
import image_analysis

#This class will be used to load data from the specified folder and
# retrieve it using the index. Maybe get an elegant method for iterating
# through it without having to load it all at once (see Generators from TF)
class DataManager(data.Dataset):
    def __init__(self, folder_path, train, use_small_patches):
        super(DataManager, self).__init__()
        print("Created Data Manager")
        self.folder_path = folder_path
        self.use_small_patches = use_small_patches
        if train:
            self.path_base = join(self.folder_path, "train")
        else:
            self.path_base = join(self.folder_path, "test")

        self.a_path = join(self.path_base, "a")
        self.b_path = join(self.path_base, "b")
        # self.generate_all_paths()

        transform_list_a = [transforms.ToTensor()]
        transform_list_b = [transforms.ToTensor()]

        self.transform_a = transforms.Compose(transform_list_a)
        self.transform_b = transforms.Compose(transform_list_b)

        self.generate_all_paths()

        self.starting_points = np.array([40, 60, 80, 100, 120, 140, 160, 180, 200, 220])


    # TODO: This might be a bad solution once folders get too big -> work in batches then or generator?
    # ARE THERE EVEN GENERATORS LIKE IN TF?
    def generate_all_paths(self):
        print("Getting image paths...")

        print("TRAIN A B:")
        self.a_images = self.get_all_imgs_in_folder(self.a_path)
        self.b_images = self.get_all_imgs_in_folder(self.b_path)
        # print(self.a_images)
        # print(self.b_images)

    #TODO: Just use a list of file endings maybe for cleaner code
    def get_all_imgs_in_folder(self, folder):
        file_paths = []
        # print("GET IMAGES IN " + folder)
        for file in listdir(folder):
            # print(file)
            # if file.endswith(".jpg") or file.endswith(".jpeg" or file.endswith(".png")
            #                          or file.endswith(".JPG") or file.endswith(".JPEG")
            #                          or file.endswith(".PNG")):
            #         print("SAVE FILE " + str(join(folder, file)))
            #         file_paths.append(join(folder, file))
            if file.__contains__(".png"):
                file_paths.append(join(folder, file))
        # print(file_paths)
        return file_paths

    def get_imgs_at_index(self, index):
        #TODO: Seems necessary in order to use halfing and doubling of the sizes properly
        # a_img = PIL.Image.open(self.a_images[index]).resize((320, 240), PIL.Image.BICUBIC)
        # b_img = PIL.Image.open(self.b_images[index]).resize((320, 240), PIL.Image.BICUBIC)

        a_img = cv2.imread(self.a_images[index])
        b_img = cv2.imread(self.b_images[index], 0) #If we want RGB output, the second param should be 1 or empty

        # a_img = cv2.resize(a_img, (0, 0), fx=0.75, fy=0.75)
        # b_img = cv2.resize(b_img, (0, 0), fx=0.75, fy=0.75)

        a_img = cv2.resize(a_img, (0, 0), fx=0.5, fy=0.5)
        b_img = cv2.resize(b_img, (0, 0), fx=0.5, fy=0.5)

        # grads = image_analysis.get_avg_gradient(b_img)
        # num_zeros = image_analysis.check_for_zeros(b_img)
        #print("IMAGE SHAPES:")
        #print(a_img.shape)

        if self.use_small_patches:
            index = np.random.choice(self.starting_points.shape[0], 1, replace=False)
            starting_point = self.starting_points[index[0]]
            print("STARTING POINT: " + str(starting_point))
            a_img = a_img[starting_point:starting_point + 64, starting_point:starting_point + 64] #replace starting_point + 64 with 286 for original size
            a_img = a_img[..., ::-1] #convert BGR (cv2) to RGB (what we want later)
            a_img = a_img.copy() #remove negative strides
    
            b_img = b_img[starting_point:starting_point + 64, starting_point:starting_point + 64]
            b_img = b_img[:, :, np.newaxis] #add the extra axis so we can work with this as  we do with rgb -> less code
        else:
            # a_img = a_img[30:286, 30:286] #TODO: This is for squares
            a_img = a_img[3:131, 50:562]
            a_img = a_img[..., ::-1]  # convert BGR (cv2) to RGB (what we want later)
            a_img = a_img.copy()  # remove negative strides

            # b_img = b_img[30:286, 30:286]
            b_img = b_img[3:131, 50:562]
            b_img = b_img[:, :, np.newaxis]  # add the extra axis so we can work with this as  we do with rgb -> less code
            
        #print(a_img.shape)

        #print("MIN AND MAX a_img: " + str(np.amin(a_img)) + "--" + str(np.amax(a_img)))
        # print("MIN AND MAX b_img: " + str(np.amin(b_img)) + "--" + str(np.amax(b_img)))

        # a_img = PIL.Image.open(self.a_images[index]).resize((480, 360), PIL.Image.BICUBIC)
        # b_img = PIL.Image.open(self.b_images[index]).resize((480, 360), PIL.Image.BICUBIC)

        # a_img = a_img.crop((30, 30, 286, 286))
        # b_img = b_img.crop((30, 30, 286, 286))

        a = self.transform_a(a_img)
        b = self.transform_b(b_img)

        # a = np.transpose(a_img, (2, 0, 1))
        # b = np.transpose(b_img, (2, 0, 1))
        #
        # a = torch.from_numpy(a)
        # b = torch.from_numpy(b)


        #TODO: Probably, we don't need 256x256
        # a = self.transform(PIL.Image.open(self.a_images[index]).crop((8, 48, 232, 272)))
        # b = self.transform(PIL.Image.open(self.b_images[index]).crop((8, 48, 232, 272)))

        # a = self.transform(PIL.Image.open(self.a_images[index]).resize((240, 320), PIL.Image.BICUBIC))
        # b = self.transform(PIL.Image.open(self.b_images[index]).resize((240, 320), PIL.Image.BICUBIC))
        return a, b

    def __getitem__(self, index):
        a, b = self.get_imgs_at_index(index)
        return a, b

    def __len__(self):
        return len(self.a_images)

