from QMetricFunctions import *
import numpy as np
from PIL import Image

# Specifying image name
IMG_NAME = '/media/udit/Backup_Files/PythonQ/testImages/I_GT.png'

img = Image.open(IMG_NAME)
arr = np.array(img)

print("PIL mode:", img.mode)
print("dtype:", arr.dtype, "shape:", arr.shape)
print("min/max:", arr.min(), arr.max())


# Measure Q
#Q = calculateQ(arr, 0.001)

#print('Q = ', Q)