#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from skimage.util import view_as_blocks
from concurrent.futures import ProcessPoolExecutor


'''
Function to calculate local metric
    input: Image Patch
    output: local metric for the patch

'''
def calculateLocalMetric(patch):
    # Calculate gradient
    gx, gy = np.gradient(patch, edge_order=2)

    G = np.column_stack((gx.ravel(), gy.ravel()))

    # Calculate SVD values
    svd_value = np.linalg.svd(G, compute_uv=False)

    s1 = svd_value[0]
    s2 = svd_value[1]

    R = (s1 - s2) / (s1 + s2 + 1e-8)

    q_metric = s1 * R

    return q_metric
'''
Function to calculate threshold
'''
#@njit
def getThreshold(delta, N):
    power_val = (N ** 2) - 1
    k_val = delta ** (1.0 / power_val)

    numerator = np.sqrt(1 - k_val)
    denominator = np.sqrt(k_val + 1)

    tau = numerator / denominator

    return tau

'''
Function to calculate local coherence
'''

def calculateLocalCoherence(patch):
    # Calculate gradient
    gx, gy = np.gradient(patch, edge_order=2)

    G = np.column_stack((gx.ravel(), gy.ravel()))

    # Calculate SVD values
    svd_value = np.linalg.svd(np.dot(G.T, G), compute_uv=False)

    s1 = svd_value[0]
    s2 = svd_value[1]

    R = (s1 - s2) / (s1 + s2 + 1e-8)

    coherence = R

    return coherence

'''
Function to calculate metric Q
'''
def calculateQ(img, delta, patch_size=8):
    # Initializing patch size as 8x8
    #patch_size = (8, 8)
    patch_size = (patch_size, patch_size)

    # Convert image to double
    #img = (img/255.0).astype(np.float64)
    img = img.astype(np.float64) / np.iinfo(img.dtype).max

    # Divide image into patches
    patches = view_as_blocks(img, patch_size)

    # Calculating dimensions of the image
    h, w = img.shape

    total_patches = (h // patch_size[0]) * (w // patch_size[1])

    # Initialize local coherence
    local_coherence = np.zeros((patches.shape[0], patches.shape[1]))

    # Generate threshold based on formula
    threshold = getThreshold(delta, patch_size[0])

    # Calculate local coherence based on formula
    with ProcessPoolExecutor() as executor:
        local_coherence = np.array(list(executor.map(calculateLocalCoherence, patches.reshape(-1, patch_size[0], patch_size[1]))))
        local_coherence = local_coherence.reshape(patches.shape[0], patches.shape[1])

    # Define type of patches - anisotropic or isotropic
    anisotropic_patches = (local_coherence.flatten() > threshold)

    # Initialize local metric for each patch
    local_metric = np.zeros(anisotropic_patches.shape)

    # Calculate local metric for each patch
    with ProcessPoolExecutor() as executor:
        local_metric[anisotropic_patches] = list(executor.map(calculateLocalMetric, patches.reshape(-1, patch_size[0], patch_size[1])[anisotropic_patches]))

    summed_value = local_metric.sum()

    local_metric = local_metric.reshape((patches.shape[0], patches.shape[1]))

    q_metric = summed_value.sum() / local_metric.size #patches.shape[0] #16

    return q_metric

def main():
    # Init parser
    parser = argparse.ArgumentParser(description="Measure Q for media files")

    # Add arguments
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("-d", "--delta", type=float, default=0.001, help="Delta for threshold calculation")
    parser.add_argument("-p", "--patch_size", type=int, default=8, help="Patch size")
    args = parser.parse_args()

    # Read the image
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    # Check if the image is divisible by the patch size
    if img.shape[0] % args.patch_size != 0 or img.shape[1] % args.patch_size != 0:
        
        # Pad image to adjust for patch size
        pad_h = (args.patch_size - (img.shape[0] % args.patch_size)) % args.patch_size
        pad_w = (args.patch_size - (img.shape[1] % args.patch_size)) % args.patch_size

        # Zero pad the image
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not read image {args.input}")
        return

    # Measure Q
    Q_value = calculateQ(img, args.delta, patch_size=args.patch_size)

    # Print out Q
    print(f"Q: {Q_value}")

if __name__ == "__main__":
    main()
