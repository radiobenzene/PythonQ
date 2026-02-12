#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from skimage.util import view_as_blocks
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import csv

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

# Function to pad the image
def PadImagePerPatch(I_gray, patch_size):

    h, w = I_gray.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size

    # Pad the image with zeros
    padded_image = cv2.copyMakeBorder(I_gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    return padded_image


def main():
    
    # Specify error codes
    ERROR_INVALID_FOLDER = -1
    ERROR_INVALID_IMAGE = -2
    ERROR_INVALID_VIDEO = -3
    ERROR_INVALID_CSV = -4

    # All image extensions
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # All video extensions
    VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"}

    # Init parser
    parser = argparse.ArgumentParser(description="Measure Q for media files")

    # Add arguments
    parser.add_argument("input", type=str, nargs="?", help="Path to the input image")
    parser.add_argument("-f", "--folder", type=str, help= "Path to folder containing images")
    parser.add_argument("-c", "--csv", type=str, help="Path to save results as a csv")
    parser.add_argument("-d", "--delta", type=float, default=0.001, help="Delta for threshold calculation")
    parser.add_argument("-p", "--patch_size", type=int, default=8, help="Patch size")
    args = parser.parse_args()

    #ext = path.suffix.lower()

    # Folder handling
    if args.folder:
        # Get path
        folder_path = Path(args.folder)

        # Check if path is actually a folder
        if not folder_path.is_dir(): 
            print(f"Error: {args.folder} is not a valid folder path")
            return ERROR_INVALID_FOLDER
        
        # Get the valid images
        image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in IMG_EXTS]
        if not image_files:
            print(f"Error: No valid image files found in folder {args.folder}")
            return ERROR_INVALID_FOLDER
        
        res = []
        overall_Q = 0.0
        counter = 0

        print(f"Processing folder: {folder_path}")
        for img_path in tqdm(image_files, desc="Processing Images", dynamic_ncols=True, unit="image"):
            
            # Read the image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            # Check if image is loaded properly
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue

            # Pad the imahge
            img = PadImagePerPatch(img, args.patch_size)

            # Measure Q
            Q_value = calculateQ(img, args.delta, patch_size=args.patch_size)

            # Store in array
            res.append((img_path.name, Q_value))
            overall_Q += Q_value
            counter += 1

        # Get average Q
        average_Q = overall_Q / counter if counter > 0 else 0.0
        print(f"Average Q for folder {folder_path}: {average_Q}")

        # Check if csv is needed
        if args.csv:
            try:
                with open(args.csv, mode='w', newline='') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(["Image", "Q"])
                    writer.writerows(res)

                    # May add average Q at the end
                    writer.writerow(["Average", average_Q])

                print(f"Results saved to {args.csv}")
            except Exception as e:
                print(f"Error: Could not write to CSV file {args.csv}. Exception: {e}")
                return ERROR_INVALID_CSV
        
        return average_Q




    if args.input:
        path = Path(args.input)
        ext = path.suffix.lower()
        # Check if the media file is an image
        if ext in IMG_EXTS:
            # Read the image
            img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

            # Check if image is loaded properly
            if img is None:
                print(f"Error: Could not read image {args.input}")
                return
            
            # Pad the image if necessary
            img = PadImagePerPatch(img, args.patch_size)

            # Measure Q
            Q_value = calculateQ(img, args.delta, patch_size=args.patch_size)

            # Print out Q
            print(Q_value)

            # Return Q
            return Q_value
        


        # Check if media file is a video
        if ext in VIDEO_EXTS:
            # Read the video
            vid = cv2.VideoCapture(args.input)

            # Get total frames
            total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = total_frames if total_frames > 0 else None
            

            # Check if video is loaded properly
            if not vid.isOpened():
                print(f"Error: Could not read video {args.input}")
                return
            
            # Initialize Q values list
            Q_vals = 0.0
            counter = 0

            # Init progress bar
            with tqdm(total=total_frames, desc="Processing Video", dynamic_ncols=True, unit="frame") as pbar:
                while True:
                    # Get frame
                    ret, frame = vid.read()

                    if not ret:
                        break

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Pad the frame
                    gray_frame = PadImagePerPatch(gray_frame, args.patch_size)

                    # Measure Q for the frame
                    Q_value = calculateQ(gray_frame, args.delta, patch_size=args.patch_size)
                    #print(f"Frame {counter}: Q = {Q_value}")

                    # Increment global Q and counter
                    Q_vals += Q_value
                    counter += 1
                    pbar.update(1)

            pbar.close()
            vid.release()

                # Get average Q over all frames
            average_Q = Q_vals / counter
            print(average_Q)

            # Return average Q
            # return average_Q
    else:
        parser.print_help()
        
            
if __name__ == "__main__":
    main()
