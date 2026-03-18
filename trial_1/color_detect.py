import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

#trial 1
""" def get_dominant_color(image_path, k=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pixels = img.reshape((-1,3))
    clstr = KMeans(n_clusters=k)
    clstr.fit(img_pixels)
    labels, counts = np.unique(clstr.labels_, return_counts = True)
    dominant_color = clstr.cluster_centers_[np.argmax(counts)]
    return dominant_color.astype(int)
color = get_dominant_color("C:/-VEES/rover_vision/balloons/green-party-balloon_25030-68250.jpg")
print(color)#[254 254 254]-gets the background """

color_dict = {(0,0,0):"black", (255,255,255):"white", (255,0,0):"red", (0,255,0):"green", (0,0,255):"blue", (255,255,0):"yellow", (255,0,255):"magenta", (0,255,255):"cyan"}

def get_color_name(rgb):
    color_names = list(color_dict.values())
    color_rgb = list(color_dict.keys())
    tree = KDTree(color_rgb)
    dist, index = tree.query(rgb)
    return color_names[index]

def get_object_color(image_path):
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

rgb, name = get_object_color("C:/-VEES/rover_vision/balloons/balloons-isolated_23-2151146053.jpg")
print(rgb, name)
    
