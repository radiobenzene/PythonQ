from QMetricFunctions import *
import numpy as np
from PIL import Image
# Specifying image name
IMG_NAME = 'barbara.bmp'

# Read image
img = Image.open(IMG_NAME)

# Convert image to YUV color 
img = img.convert('YCbCr')

# Extract only Y channel
img = img.split()[0]

# Convert to numpy array
img_np = np.array(img)

# Measure Q
Q = calculateQ(img_np, 0.001)

print('Q = ', Q)