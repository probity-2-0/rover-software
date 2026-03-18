"""
Description:This file contains code to that detects the dominant color (or highest frequency color) in an image. It is quite inefffective since it is based on identifying the dominant color in the image, it often detects the background color instead of the object color. 
--
Author: Amy Kibara
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

#small color dictionary with RGB values and their corresponding color names. This will be used to match the detected color to a known color name.
color_dict = {(0,0,0):"black", (255,255,255):"white", (255,0,0):"red", (0,255,0):"green", (0,0,255):"blue", (255,255,0):"yellow", (255,0,255):"magenta", (0,255,255):"cyan"}
#ensure your file path is correct and the image exists at that location, otherwise it will return an error. 
img_path = "C:/-VEES/peeps/rover-software/four_balloons_project/pics/green_balloon.jpg"

def get_color_name(rgb):
    """ This function takes an RGB color as input and returns the name of the closest color from the color dictionary. It uses a KDTree for efficient nearest neighbor search."""
    color_names = list(color_dict.values())
    color_rgb = list(color_dict.keys())
    tree = KDTree(color_rgb)
    dist, index = tree.query(rgb)
    return color_names[index]

def get_dominant_color(image_path, k=3):
    """ This function takes an image path as input and returns the RGB values of the dominant color. It reads the image, converts it to RGB format, reshapes it into a 2D array of pixels, and applies KMeans clustering to find the dominant color."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pixels = img.reshape((-1,3))
    clstr = KMeans(n_clusters=k)
    clstr.fit(img_pixels)
    labels, counts = np.unique(clstr.labels_, return_counts = True)
    dominant_color = clstr.cluster_centers_[np.argmax(counts)]
    return dominant_color.astype(int)


color = get_dominant_color(img_path)
name = get_color_name(color)
print(color, name) #[254 254 254] white -gets the background; unwanted result.

""" 
Possible suggestions to improve the code:
1. Use a larger color dictionary with more colors and their corresponding RGB values to improve the accuracy of color detection.
2. Use a different clustering algorithm, such as DBSCAN or Mean Shift, which may be better suited for color detection in images with varying lighting conditions and backgrounds.
3. Implement a pre-processing step to remove the background from the image before applying color detection, such as using a segmentation algorithm or a color thresholding technique.
4. Use a different color space, such as HSV or LAB, which may be more effective for color detection in images with varying lighting conditions and backgrounds.
5. Implement a post-processing step to filter out noise and improve the accuracy of color detection, such as using morphological operations or a median filter.
6. Use a larger dataset of images with known colors to train a machine learning model for color classification, which may improve the accuracy of color detection in new images.
7. Implement a confidence score for the detected color, which can help to determine the reliability of the color detection results and allow for better handling of cases where the detected color is uncertain.
8. Use a different distance metric, such as Euclidean distance or cosine similarity, to compare the detected color with the colors in the color dictionary, which may improve the accuracy of color matching.
10. Use a different method for determining the dominant color, such as calculating the histogram of colors in the image and selecting the color with the highest frequency, which may be more effective for images with multiple colors and varying lighting conditions.
 """