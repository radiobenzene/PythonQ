from QMetricFunctions import *
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

# Image Folder Path
TEST_FOLDER = Path("testImages")

# Specify delta
DELTA = 0.001

# Check list of supported extensions
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for p in sorted(TEST_FOLDER.iterdir()):
    if p.suffix.lower() not in exts:
        continue
    
    # Read the image in grayscale
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        continue

    # Measure Q
    q = calculateQ(img, DELTA)

    # As sanity, print out Q
    print(f"{p.name}\t{q}")