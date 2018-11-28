import numpy as np
from PIL import Image

#atm just for saving the image but could use it for loading etc as well

def save_tensor_as_image(image_tensor, filename):
    #get numpy array from tensor
    img_array = image_tensor.float().numpy()
    img_array = (np.transpose(img_array, (1, 2, 0)) + 1) / 2.0 * 255.0  # reformat for conversion

    if img_array.shape[2] != 1:
        print("SHAPE: " + str(img_array.shape))
        img_array = img_array.astype(np.uint8)

        print(img_array.shape)
        #generate pillow image and save it
        img = Image.fromarray(img_array)
        img.save(filename)

        print("saved" + filename)
    else:
        # img_array = img_array[:, :, 0] #For black-white imgs, we need a 2d matrix

        img = Image.fromarray(img_array, mode='L')
        img.save(filename)

        print("SAVED " + filename)