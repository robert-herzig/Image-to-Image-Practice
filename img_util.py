import numpy as np
from PIL import Image

#atm just for saving the image but could use it for loading etc as well

def save_tensor_as_image(image_tensor, filename):
    #get numpy array from tensor
    img_array = image_tensor.float().numpy()
    img_array = (np.transpose(img_array, (1, 2, 0)) + 1) / 2.0 * 255.0 #reformat for conversion
    img_array = img_array.astype(np.uint8)

    #generate pillow image and save it
    img = Image.fromarray(img_array)
    img.save(filename)

    print("saved" + filename)