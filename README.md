# Python Implementation of $Q$

Unofficial Python implementation of the sharpness metric $Q$ proposed by Zhu and Milanfar - [https://ieeexplore.ieee.org/abstract/document/5484579]

Check MATLAB implementation for better understanding nuances - [https://github.com/radiobenzene/MetricQ]

# Creating the environment
To recreate the conda environment, you may use either of the commands.

`pip install -r requirements.txt`

`conda env create -f environment.yml`

Using the `.yml` approach will result in the generation of a conda environment with the name `metricQ_official`. Therefore, to rename the conda environment generated using this approach with the name `foo`, we recommend using the following command.

`conda env create -f environment.yml -m foo`

# Brief Description
$Q$ is a sharpness metric measured only the Luma (Y) channel of an image. A higher value corresponds to a sharper image, whereas a lower one corresponds to a blurry or noisy image.

# Using $Q$
`QMetricFunctions.py` is a file that consists of functions required for calculating the metric. `sampleUsage.oy` is a script that demonstartes how to use the metric for different images. 

`Q.py` is a CLI for measuring the metric on 
- A single image
- A folder of images
- A video sequence

For running the metric on a single image - `img.png` with a patch size `p_size` (default patch size is 8), use the following command.

`python3 Q.py img.png -p p_size`

The delta value is set to 0.001. To change this, to 0.005, for instance, use the following command.

`python3 Q.py img.png -d 0.005`

For running the metric on images in a folder - `folder1`, use the following command.

`python3 Q.py -f folder1`
 
 This command returns the average Q for all images in that folder.

 To generate a csv, named `test.csv` for images in the folder `folder1` with two columns, viz. ImageName in column 1 and $Q$ in column 2, use the following command

`python3 Q.py -f folder1 -c test.csv`

For running the metric on a video sequence - `video.mp4`, use the following command.

`python3 Q.py video.mp4`

To visualize sharpness of each frame for the video sequence `video.mp4`, use the following command.

`python3 Q.py video.mp4 -r`

This will plot $Q$ frame-wise for the video sequence. 


# Notes
For an HD footage, use a patch size of 32 or above. 

# Useful Links
- [MATLAB Implementation](https://github.com/radiobenzene/MetricQ)
- [Metric $Q$ Paper](https://ieeexplore.ieee.org/abstract/document/5484579)
- [Metric $Q$ As a Loss for Deblurring](https://github.com/aurangau/MMSP2024)
- [Tensorflow Implementation](https://github.com/aurangau/QSharpNet/blob/main/correctedQ_TF.py)
