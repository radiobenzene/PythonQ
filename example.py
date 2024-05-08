from pythonq import *
import numpy as np
from PIL import Image
import cv2
import time
# read in image
IMG_NAME = 'barbara.bmp'
img = cv2.cvtColor(cv2.imread(IMG_NAME, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB)

# extract only luma channel
img = img[:,:,0]

# add gaussian blur
blur_img_1 = cv2.GaussianBlur(img, (23, 23), sigmaX=1.5)
blur_img_2 = cv2.GaussianBlur(img, (23, 23), sigmaX=2.5)
blur_img_3 = cv2.GaussianBlur(img, (23, 23), sigmaX=3.5)

# measure q metric
start = time.time()
q_orig = calculate_q(img)
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

print('Q_orig =', round(q_orig,2))
print('Q_blur_1 =', round(q_blur_1,2))
print('Q_blur_2 =', round(q_blur_2,2))
print('Q_blur_3 =', round(q_blur_3,2))