import numpy as np
from skimage.util import view_as_blocks
import cv2
from concurrent.futures import ProcessPoolExecutor
EPS = np.finfo(np.float64).eps

'''
    get threshold:
    
    Defining the null hypothesis H0 as: “The given patch is isotropic with white Gaussian noise,” we can calculate the 
    metric and use it as a test statistics to decide whether to reject the null hypothesis H0. Numerically, the test is 
    carried out by comparing the calculated for the patch to a preselected threshold.
'''

def get_threshold(delta, N):
    k_val = delta ** (1.0 / (N ** 2 - 1))

    numerator = np.sqrt(1 - k_val)
    denominator = np.sqrt(k_val + 1)

    return numerator / denominator

'''
    calculate_patch_q:
    
    Calculates the q metric for an individual patch from an image. This is done by calculating the singular values for the gradient
    of a patch and then calculating the coherence of the patch. If the coherence is greater than the threshold, the q metric is the 
    patches coherence multiplied by it's dominant singular value. Patches with a coherence less than the threshold have a q metric 
    of 0 (isotropic patch).
'''

def calculate_patch_q(args):
    patch, threshold = args
    
    # calculate gradient
    gx, gy = np.gradient(patch)
    gx = gx.ravel()
    gy = gy.ravel()
    G = np.column_stack((gx, gy))

    # calculate svd values
    s = np.linalg.svd(G, compute_uv=False, full_matrices=False)
    
    # calculate coherence
    R = (s[0] - s[1]) / (s[0] + s[1] + EPS)
    
    # calculate q metric
    if R > threshold:
        return s[0] * R
    else:
        return 0

'''
    calculate_q:
    
    Calculates the Q-metric for an image to determine blur magnitude, based on the work of Milanfar et al.
    https://ieeexplore.ieee.org/abstract/document/5484579
'''
def calculate_q(img, delta=0.001, patch_size=(8, 8)):
    img = (img / 255.0).astype(np.float64)

    # pad image to make dimensions divisible by patch size
    pad_height = patch_size[0] - img.shape[0] % patch_size[0]
    pad_width = patch_size[1] - img.shape[1] % patch_size[1]
    img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant')
    
    # divide image into patches
    patches = view_as_blocks(img, patch_size)
    local_q = np.zeros((patches.shape[0], patches.shape[1]))
    total_patches = (patches.shape[0] // patch_size[0]) * (patches.shape[1] // patch_size[1])
    
    threshold = get_threshold(delta, patch_size[0])
    patches_reshaped = patches.reshape(-1, patch_size[0], patch_size[1])
    threshold_list = [threshold] * patches_reshaped.shape[0]

    # calculate local coherence based on formula
    with ProcessPoolExecutor() as executor:
        local_q = np.array(list(executor.map(calculate_patch_q, zip(patches_reshaped, threshold_list))))

    q_metric = np.sum(local_q) / patches.shape[1]

    return q_metric