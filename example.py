from pythonq import *
import numpy as np
from PIL import Image
import cv2
import time
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std_dev=1):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # ensure valid pixel ranges
    return noisy_image.astype(np.uint8)

# read in image
IMG_NAME = 'barbara.bmp'
orig_img = cv2.cvtColor(cv2.imread(IMG_NAME, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB)

# extract only luma channel
orig_img = orig_img[:,:,0]

# add gaussian blur
blur_img_1 = cv2.GaussianBlur(orig_img, (23, 23), sigmaX=1.5)
blur_img_2 = cv2.GaussianBlur(orig_img, (23, 23), sigmaX=2.5)
blur_img_3 = cv2.GaussianBlur(orig_img, (23, 23), sigmaX=3.5)

# add gaussian noise
noisy_img_1 = add_gaussian_noise(orig_img, std_dev=10)
noisy_img_2 = add_gaussian_noise(orig_img, std_dev=20)
noisy_img_3 = add_gaussian_noise(orig_img, std_dev=30)

# measure q metric for blurred images
start = time.time()
q_orig = calculate_q(orig_img)
end = time.time()
print("Calculating Q for unblurred image took ",end - start, " seconds")

start = time.time()
q_blur_1 = calculate_q(blur_img_1)
end = time.time()
print("Calculating Q for blurred image took ",end - start, " seconds")

start = time.time()
q_blur_2 = calculate_q(blur_img_2)
end = time.time()
print("Calculating Q for blurred image took ",end - start, " seconds")

start = time.time()
q_blur_3 = calculate_q(blur_img_3)
end = time.time()
print("Calculating Q for blurred image took ",end - start, " seconds")

# measure q metric for noisy images
start = time.time()
q_noisy_1 = calculate_q(noisy_img_1)
end = time.time()
print("Calculating Q for noisy image took ",end - start, " seconds")

start = time.time()
q_noisy_2 = calculate_q(noisy_img_2)
end = time.time()
print("Calculating Q for noisy image took ",end - start, " seconds")

start = time.time()
q_noisy_3 = calculate_q(noisy_img_3)
end = time.time()
print("Calculating Q for noisy image took ",end - start, " seconds")

images = [orig_img, blur_img_1, blur_img_2, blur_img_3]
q_values = [q_orig, q_blur_1, q_blur_2, q_blur_3]
sigmas = ["None", "1.5", "2.5", "3.5"]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax, img, q, sigma in zip(axs.ravel(), images, q_values, sigmas):
    ax.imshow(img, cmap='gray')
    ax.set_title(r'$Q = {}, \sigma_x = {}$'.format(round(q, 2), sigma))
    ax.axis('off')

plt.tight_layout()
plt.savefig('q_metric_blur.png')

images = [orig_img, noisy_img_1, noisy_img_2, noisy_img_3]
q_values = [q_orig, q_noisy_1, q_noisy_2, q_noisy_3]
sigmas = ["None", "10", "20", "30"]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax, img, q, sigma in zip(axs.ravel(), images, q_values, sigmas):
    ax.imshow(img, cmap='gray')
    ax.set_title(r'$Q = {}, \sigma = {}$'.format(round(q, 2), sigma))
    ax.axis('off')

plt.tight_layout()
plt.savefig('q_metric_noise.png')