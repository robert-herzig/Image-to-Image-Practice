import numpy as np
import cv2


def get_avg_gradient(image):
    print("GETTING GRADIENTS...")
    img = np.array(image)
    print("SHAPE FOR GRADIENTS: " + str(img.shape))
    grads = np.gradient(img)
    avg_grad_rows = np.average(grads, axis=1)
    avg_grad = np.average(avg_grad_rows, axis=0)
    avg_grad = np.average(avg_grad, axis=0)
    print(avg_grad)
    return avg_grad

def check_for_zeros(image):
    print("COUNTING ZEROS...")
    img = np.array(image)
    num_zeros = np.count_nonzero(img == 0)
    total_pixels = img.shape[0] * img.shape[1]
    relative_zeros = num_zeros / total_pixels
    print("RELATIVE AMOUNT ZEROS: " + str(relative_zeros))

    return relative_zeros

