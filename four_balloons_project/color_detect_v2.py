""" 
Description: This file contains code to detect the color of an object in an image. It uses KMeans clustering to find the dominant color in the image and then matches it to a predefined set of colors using a KDTree for efficient nearest neighbor search. The function get_object_color takes an image path as input and returns the RGB values and the name of the detected color. 
--
Author: Amy Kibara
 """

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

#small color dictionary with RGB values and their corresponding color names. This will be used to match the detected color to a known color name.
color_dict = {(0,0,0):"black", (255,255,255):"white", (255,0,0):"red", (0,255,0):"green", (0,0,255):"blue", (255,255,0):"yellow", (255,0,255):"magenta", (0,255,255):"cyan"}
img_path = "C:/-VEES/peeps/rover-software/four_balloons_project/pics/green_balloon.jpg"

def get_color_name(rgb):
    """ This function takes an RGB color as input and returns the name of the closest color from the color dictionary. It uses a KDTree for efficient nearest neighbor search."""

    color_names = list(color_dict.values())
    color_rgb = list(color_dict.keys())
    tree = KDTree(color_rgb)
    dist, index = tree.query(rgb)
    return color_names[index]

def get_object_color(image_path):
    """ This function takes an image path as input and returns the RGB values and the name of the detected color. It reads the image, converts it to RGB format, reshapes it into a 2D array of pixels, and applies a mask to filter out pixels that are close to white (background). It then uses KMeans clustering to find the dominant color among the remaining pixels and matches it to a known color name using the get_color_name function."""

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pixels = img.reshape((-1,3))
    mask = np.any(img_pixels<240, axis=1)
    object_pixels = img_pixels[mask]
    if len(object_pixels)==0:
        return "all-white image"
    clstr = KMeans(n_clusters=2, n_init=10)
    clstr.fit(object_pixels)
    labels, counts = np.unique(clstr.labels_, return_counts = True)
    dominant_rgb = clstr.cluster_centers_[np.argmax(counts)].astype(int)
    color_name = get_color_name(dominant_rgb)
    return dominant_rgb, color_name

#example usage
rgb, name = get_object_color(img_path)
print(rgb, name)
    
