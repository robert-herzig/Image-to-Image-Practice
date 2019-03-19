import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.utils
import cv2
# import png
# import skimage.io
#atm just for saving the image but could use it for loading etc as well

def save_tensor_as_image(image_tensor, filename):
    #get numpy array from tensor
    image_tensor = image_tensor.cpu()
    img_array = image_tensor.float().numpy()
    # img_array = (np.transpose(img_array, (1, 2, 0)) + 1) / 2.0 * 255.0  # reformat for conversion
    img_array = (np.transpose(img_array, (1, 2, 0)) * 255.0)

    if img_array.shape[2] != 1:
        #print("SHAPE: " + str(img_array.shape))
        img_array = img_array.astype(np.uint8)

        # print(img_array.shape)
        #generate pillow image and save it
        img = Image.fromarray(img_array)
        img.save(filename)

        #print("SAVED (RGB) " + filename)
    else:
        img_array = img_array[:, :, 0] #For black-white imgs, we need a 2d matrix
        # img_array = np.interp(img_array, (127.5, 255), (0, 255))
        #print("MIN / MAX OF OUTPUT: " + str(np.amin(img_array)) + "--" + str(np.amax(img_array)))

        # print(np.amax(img_array))
        # print(np.amin(img_array))
        cv2.imwrite(filename, img_array)
        #
        # img = Image.fromarray(img_array, mode='L')
        # img.save(filename)

        #torchvision.utils.save_image(image_tensor, filename, range=(-1, 1))
        # image = F.to_pil_image(image_tensor)
        # image.save(filename, )
        #skimage.io.imsave(filename, img_array)
        # png.fromarray(img_array, 'L').save(filename)

        #print("SAVED (BW) " + filename)
