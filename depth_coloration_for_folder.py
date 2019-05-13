import numpy as np
from depth_map_visualization import color_depth_map
import os
import cv2

class DepthColoration:
    def __init__(self):
        print("INIT DEPTH COL")

    def colorize_all_in_folder(self, folder):
        for filename in os.listdir(folder):
            if not filename.__contains__(".png"):
                continue
            full_path = os.path.join(folder, filename)
            gray = cv2.imread(full_path, 0)
            col = color_depth_map(gray)
            cv2.imwrite(full_path[:-4] + "col.png", col)





if __name__ == '__main__':
    target_folder = "C:\\Users\\Rob\\Desktop\\MASSH\\EXPERIMENTS_PRESENTATION\\RecolorizationTest"
    dcol = DepthColoration()
    dcol.colorize_all_in_folder(target_folder)
