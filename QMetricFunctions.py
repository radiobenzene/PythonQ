import numpy as np
from skimage.util import view_as_blocks
import cv2
from concurrent.futures import ProcessPoolExecutor

'''
Function to calculate local metric
'''

def calculateLocalMetric(patch):
    # Calculate gradient
    gx, gy = np.gradient(patch)

    gx = gx.ravel()
    gy = gy.ravel()
    G = np.column_stack((gx, gy))

    # Calculate SVD values
    svd_value = np.linalg.svd(G, compute_uv=False)

    s1 = svd_value[0]
    s2 = svd_value[1]

    R = (s1 - s2) / (s1 + s2 + 1e-6)

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
    gx, gy = np.gradient(patch)

    gx = gx.ravel()
    gy = gy.ravel()
    G = np.column_stack((gx, gy))

    # Calculate SVD values
    svd_value = np.linalg.svd(np.dot(G.T, G), compute_uv=False)

    s1 = svd_value[0]
    s2 = svd_value[1]

    R = (s1 - s2) / (s1 + s2 + np.finfo(np.float64).eps)

    coherence = R

    return coherence

'''
Function to calculate metric Q
'''
def calculateQ(img, delta):
    # Initializing patch size as 8x8
    patch_size = (8, 8)

    # Convert image to double
    img = (img/255.0).astype(np.float64)

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

    q_metric = summed_value.sum() / patches.shape[0] #16

    return q_metric