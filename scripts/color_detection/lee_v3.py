import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
#How can we find the color boundaries ofcolors where we have no  clear RGB in an image?
#Need a means to find this boundaries so that we can perform color detection on such an image

#Reading and displaying the image
img = cv2.imread(r"C:\Users\PC\Downloads\kmeanssample.webp")
h, w, d = img.shape
print(f"Height : {h}, Width : {w}, Depth : {d}")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

#using K-Means Clustering to identify the dominant colors in the image
#this clustering can  automatically identify dominant colors in an image and then the colors I get I can use them to define BGR ranges
#Try Learning the background of how exactly this algorithm works

from sklearn.cluster import KMeans

def return_dominant_colors(image, k = 4):#what does y
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dominant_colors = return_dominant_colors(img_rgb, 4)

# Display dominant colors
plt.figure(figsize=(10, 2))#What does this line do
for i, color in enumerate(dominant_colors):
    plt.fill_between([i, i+1], 0, 1, color=color/255)
plt.axis("off")
plt.title("Dominant Colors")
plt.show()
#Continue tomorror