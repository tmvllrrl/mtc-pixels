import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import random

RGB_MAX = 255
HSV_H_MAX = 180
HSV_SV_MAX = 255
YUV_MAX = 255

# level values
BLUR_LVL = [7, 17, 37, 67, 107]
NOISE_LVL = [20, 50, 100, 150, 200]
DIST_LVL = [1, 10, 50, 200, 500]
RGB_LVL = [0.02, 0.2, 0.5, 0.65]

IMG_WIDTH = 200
IMG_HEIGHT = 66

KSIZE_MIN = 0.1
KSIZE_MAX = 3.8
NOISE_MIN = 0.1
NOISE_MAX = 4.6
DISTORT_MIN = -2.30258509299
DISTORT_MAX = 5.3
COLOR_SCALE = 0.25



def add_noise(image, sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = np.float32(noisy)
    return noisy

def generate_noise_image(image, noise_level=20):
    image = add_noise(image, noise_level)

    return image

def generate_blur_image(image, blur_level=7):
    image = cv2.GaussianBlur(image, (blur_level, blur_level), 0)

    return image

def generate_distort_image(image, distort_level=1):
    K = np.eye(3)*1000
    K[0,2] = image.shape[1]/2
    K[1,2] = image.shape[0]/2
    K[2,2] = 1

    image = cv2.undistort(image, K, np.array([distort_level,distort_level,0,0]))

    return image

def generate_RGB_image(image, channel, direction, dist_ratio=0.25):
    color_str_dic = {
        0: "R",
        1: "G", 
        2: "B"
    }
                   
    if direction == 4: # lower the channel value
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (0 * dist_ratio)
    else: # raise the channel value
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (RGB_MAX * dist_ratio)

    return image

def perturb_r(image, dist_ratio):
    return generate_RGB_image(image, 0, 4 if random.random() < 0.5 else 5, dist_ratio)

def perturb_g(image, dist_ratio):
    return generate_RGB_image(image, 1, 4 if random.random() < 0.5 else 5, dist_ratio)

def perturb_b(image, dist_ratio):
    return generate_RGB_image(image, 2, 4 if random.random() < 0.5 else 5, dist_ratio)

def generate_HSV_image(image, channel, direction, dist_ratio=0.25):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    color_str_dic = {
        0: "H",
        1: "S", 
        2: "V"
    }           

    max_val = HSV_SV_MAX
    if channel == 0:
        max_val = HSV_H_MAX

    if direction == 4:
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio))
    if direction == 5:
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (max_val * dist_ratio)

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image

def perturb_h(image, dist_ratio):
    return generate_HSV_image(image, 0, 4 if random.random() < 0.5 else 5, dist_ratio)

def perturb_s(image, dist_ratio):
    return generate_HSV_image(image, 1, 4 if random.random() < 0.5 else 5, dist_ratio)

def perturb_v(image, dist_ratio):
    return generate_HSV_image(image, 2, 4 if random.random() < 0.5 else 5, dist_ratio)

# FUNCTIONS FOR GENERATING THE 15 BASE PERTURBATIONS
def perturb_r_low(image, dist_ratio):
    return generate_RGB_image(image, 0, 4, dist_ratio)

def perturb_r_high(image, dist_ratio):
    return generate_RGB_image(image, 0, 5, dist_ratio)

def perturb_g_low(image, dist_ratio):
    return generate_RGB_image(image, 1, 4, dist_ratio)

def perturb_g_high(image, dist_ratio):
    return generate_RGB_image(image, 1, 5, dist_ratio)

def perturb_b_low(image, dist_ratio):
    return generate_RGB_image(image, 2, 4, dist_ratio)

def perturb_b_high(image, dist_ratio):
    return generate_RGB_image(image, 2, 5, dist_ratio)

def perturb_h_low(image, dist_ratio):
    return generate_HSV_image(image, 0, 4, dist_ratio)

def perturb_h_high(image, dist_ratio):
    return generate_HSV_image(image, 0, 5, dist_ratio)

def perturb_s_low(image, dist_ratio):
    return generate_HSV_image(image, 1, 4, dist_ratio)

def perturb_s_high(image, dist_ratio):
    return generate_HSV_image(image, 1, 5, dist_ratio)

def perturb_v_low(image, dist_ratio):
    return generate_HSV_image(image, 2, 4, dist_ratio)

def perturb_v_high(image, dist_ratio):
    return generate_HSV_image(image, 2, 5, dist_ratio)

def perturb_blur(image, dist_ratio):
    blur_level = int(dist_ratio * (107 - 7) + 7)
    if blur_level % 2 == 0: # blur has to be an odd number
        blur_level += 1
    
    return generate_blur_image(image, blur_level)

def perturb_noise(image, dist_ratio):
    noise_level = int(dist_ratio * (200 - 20) + 20)
    return generate_noise_image(image, noise_level)

def perturb_distort(image, dist_ratio):
    distort_level = int(dist_ratio * (500 - 1) + 1)
    return generate_distort_image(image, distort_level)

# MAIN FUNCTION
def generate_perturb_img(img, time_counter):

    # Original set of perturbations
    methods = [perturb_r_low, perturb_r_high, perturb_b_low, perturb_b_high, perturb_g_low, perturb_g_high, 
                perturb_h_low, perturb_h_high, perturb_s_low, perturb_s_high, perturb_v_low, perturb_v_high,
                perturb_blur, perturb_noise, perturb_distort]
    
    i = time_counter % len(methods) # time_counter % 15

    intensity = np.random.uniform(low=0.0, high=1) # random intensity
    aug_img = np.uint8(methods[i](img.copy(), intensity)) 

    return aug_img