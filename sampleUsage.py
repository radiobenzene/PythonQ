from QMetricFunctions import *
import numpy as np
from PIL import Image

# Specifying image name
IMG_NAME = '/media/udit/Backup_Files/PythonQ/testImages/I_GT.png'

# img = Image.open(IMG_NAME)
# arr = np.array(img)

img = cv2.imread(IMG_NAME, cv2.IMREAD_GRAYSCALE)


# Measure Q
Q = calculateQ(img, 0.001)

print('Q = ', Q)