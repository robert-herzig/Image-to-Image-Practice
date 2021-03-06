import h5py
from PIL import Image
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.ImageOps
from os import listdir

class MatReader:
    def __init__(self, path):
        print("WOW, READING MATS TIME?!")
        self.path = path
        self.data = h5py.File('datasets/nyu_depth_v2_labeled.mat', 'r')

    def read_mat(self):
        # with h5py.File('datasets/NYU2/nyu_depth_v2_labeled.mat', 'r') as f:
        keys = self.data.keys()
        for k in keys:
            print(k)

    def get_index(self, index):
        print("GET INDEX")
        return self.data[index]


    def get_hot(self, gray):
        cm_hot = mpl.cm.get_cmap('inferno')
        img_src = Image.fromarray(gray).convert('L')
        # img_src = gray.convert('L')
        # img_src.thumbnail((256, 512))
        im = np.array(img_src) / 6
        im = cm_hot(im)
        im = np.uint8(im * 255)
        # im = Image.fromarray(im[:, :, :4])
        # im.save('test_hot.png')
        return im[:, :, :4]

    def resize_image(self, img, wpercent):
        basewidth = 240
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return img

    # def resize_image_bw(self, img, wpercent):
    #     basewidth = 240
    #     wpercent = (basewidth / float(img.size[0]))
    #     hsize = int((float(img.size[1]) * float(wpercent)))
    #     img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    #     return img

    def convert_whole_bw_folder_to_hot(self, folder):

        for file in listdir(folder):
            counter = 0
            if file.endswith(".jpg") or file.endswith(".jpeg" or file.endswith(".png")
                                                      or file.endswith(".JPG") or file.endswith(".JPEG")
                                                      or file.endswith(".PNG")):
                counter += 1
                img = Image.open(file)
                img = img.convert('L')
                img = self.get_hot(img)
                img.save(folder + "/converted" + str(counter) + ".jpg")



if __name__ == '__main__':
    convert_to_rgb = False
    reader = MatReader("datasets/NYU2/nyu_depth_v2_labeled.mat")
    reader.read_mat()
    rgb_test = reader.get_index('images')
    depth_test = reader.get_index('depths')

    print(len(rgb_test))

    for i in range(0, len(rgb_test)):
        print(str(i) + "/" + str(len(rgb_test)))
        train_or_test = "train"
        if i % 10 == 0:
            train_or_test = "test"
        single_rgb = rgb_test[i]
        single_depth = depth_test[i]
        # single_depth = np.array(single_depth)

        single_rgb = np.moveaxis(single_rgb, -1, 0)
        single_rgb = np.moveaxis(single_rgb, -1, 0)

        img_rgb = Image.fromarray(single_rgb)
        img_rgb = reader.resize_image(img_rgb, 240)
        print("SIZE AFTER RESIZING: " + str(img_rgb.size))
        img_rgb.save("datasets/NYU2/" + train_or_test + "/a/" + str(i) + ".jpg")

        print("DEPTH SHAPE: " + str(single_depth.shape))

        if convert_to_rgb:
            pseudo_colored_depth = reader.get_hot(single_depth)
            depth_rgb = Image.fromarray(pseudo_colored_depth)
            depth_rgb = reader.resize_image(depth_rgb, 240)
            depth_rgb = depth_rgb.convert('RGB')
            depth_rgb.save("datasets/NYU2/" + train_or_test + "/b/" + str(i) + ".jpg")
        else:

            depth_img = np.array(single_depth) / 6
            depth_img = np.uint8(depth_img * 255)
            # print(depth_img)

            depth_img = Image.fromarray(single_depth)
            depth_img = depth_img.convert('L')

            # depth_img = PIL.ImageOps.invert(depth_img)

            print("BEFORE RESIZE DEPTH: " + str(single_depth.shape))
            depth_img = reader.resize_image(depth_img, 240)
            test_rgb = False
            if test_rgb:
                #TEST
                depth_rgb_test = np.array(depth_img)
                pseudo_colored_depth = reader.get_hot(single_depth)
                depth_rgb = Image.fromarray(pseudo_colored_depth)
                depth_rgb = reader.resize_image(depth_rgb, 240)
                depth_rgb = depth_rgb.convert('RGB')
                depth_rgb.save("datasets/NYU2/" + train_or_test + "/b/" + str(i) + ".jpg")
            else:
                depth_img.save("datasets/NYU2/" + train_or_test + "/b/" + str(i) + ".jpg")


