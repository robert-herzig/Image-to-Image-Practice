import cv2
import os
from PIL import Image
import numpy as np

class ImageConverter:
    def __init__(self, root_path, target_path):
        self.root_path = root_path
        self.target_path = target_path
        self.ensure_dir(self.target_path)

    def change_root(self, new_root_path):
        self.root_path = new_root_path

    def change_target(self, new_target_path):
        self.target_path = new_target_path

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def convert_all_images_in_root(self):
        target_a = os.path.join(self.target_path, "a\\")
        target_b = os.path.join(self.target_path, "b\\")
        print("A PATH = " + target_a)
        self.ensure_dir(target_a)
        self.ensure_dir(target_b)

        a_counter = 0
        b_counter = 0
        for file in os.listdir(self.root_path):
            print(file)
            if file.endswith(".ppm"):
                a_counter += 1
                full_path = os.path.join(self.root_path, file)
                raw_img = cv2.imread(full_path)
                target_path = os.path.join(target_a, str(a_counter) + ".png")
                cv2.imwrite(target_path, raw_img)

            elif file.endswith(".pgm"):
                b_counter += 1
                full_path = os.path.join(self.root_path, file)
                raw_img = cv2.imread(full_path)
                raw_img[raw_img > 250] = 0
                target_path = os.path.join(target_b, str(b_counter) + ".png")
                cv2.imwrite(target_path, raw_img)





if __name__ == '__main__':

    root_path = "D:\\MA_Data\\dinette_0001"

    dirname = os.path.dirname(__file__)
    target_path = os.path.join(dirname, 'datasets\\NYU_FROM_RAW')

    loader = ImageConverter(root_path, target_path)
    loader.convert_all_images_in_root()