import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
#How can we find the color boundaries of colors where we have no  clear RGB in an image?
#Need a means to find this boundaries so that we can perform color detection on such an image

#Reading and displaying the image
img = cv2.imread(r"C:\Root\Projects\probity\images\colorful_landscape.jpeg")
h, w, d = img.shape
print(f"Height : {h}, Width : {w}, Depth : {d}")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

#using K-Means Clustering to identify the dominant colors in the image
#this clustering can  automatically identify dominant colors in an image and then the colors I get I can use them to define BGR ranges
#Try Learning the background of how exactly this algorithm works

from sklearn.cluster import KMeans

def return_dominant_colors(image, k = 4):#By now I know the image input parameter is supposed to be a NumPy array(A color image of size 100 x 200 pixels will be repped as an array of shape (100, 200, 3), 100 rows/height 200 columns/width 3 channels RGB). Asking KMeans to find 4 clusters(dominant colors); I can change this depending on how many distinct colors I want to extract but k=4 is the default.
    pixels = image.reshape(-1, 3)#reshape the image into a 2D array; each row is a pixel and each column is a color channel(RGB)[-1 tells Numpy to figure out the correct number of rows automatically. 3 means each pixel(each row) has 3 values RGB. This means that if the sample image I have has 100 x 100 pixels, this line turns it into a 10,000 x 3 array]
    #The image is already an array but I reshape it because Kmeans expects a 2D datasets: rows = samples, columns = features. Each pixel is one sample and each pixel has 3 features(R,G,B). That is why we reshape (height,width,3) into (height*width, 3). Remember the Table samples in AI coursework
    kmeans = KMeans(n_clusters = k)#Create a Kmeans object from scikitlearn and tell kmeans how many groups/clusters you want to find and each cluster will rep one dominant colour
    kmeans.fit(pixels)#Run the KMeans algorithm on pixel data and it groups similar colors together into k clusters and each cluster has a center point(The average colour of that group)
    #So after fitting, KMeans calculates the center of ech cluster and these centers are stored in kmeans.cluster_centers_. It is a NumPy array not a list like I originally thought and its shape is (k, 3)[K rows, one eow per cluster]

    dominant_colors = kmeans.cluster_centers_.astype(int)#Gives me the RGB values of the cluster centers(The dominant colors) and then converts those values into integers since pixel values must be integers between.
    return dominant_colors#Send the list of dominant colors; the result is an array of shape(k, 3) -> k colors each with 3 values (RGB)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dominant_colors = return_dominant_colors(img_rgb, 4)#remember I set up k=4 at the function's definition so that when we call it, like we are now, and we do not explicitly provide a value for k python will automatically use 4 as the default

# Display dominant colors
plt.figure(figsize=(10, 2))#Set size of the figure in inches; width of 10 and height of 2(Ensure my color swatches are displayed in a long horizontal strip )


for i, color in enumerate(dominant_colors):# loop through each color and give both the index i and the color values color
    plt.fill_between([i, i+1], 0, 1, color=color/255)# draw a rectangle from i to i+1 filled with that color. the color/255 scales RGB values from 0-255 to 0-1 which is what Matplotlib expects
plt.axis("off")
plt.title("Dominant Colors")
plt.show()
#So I have Identified the 4 most dominant colors in the image and with this info, we will now be able to get the boundary ranges we need

#Extracting the color boundaries and plotting the color plots for vatious colors
#These Dominant colors are just single points (eg [120,200, 150]). However in real images colors vary slightly due to  lighting differences, shadows, cam noise etc so if we only look at the exact RGBvalue, you will miss most of the pixels that are close
#That is why I need to define boundary ranges around each dominant color and these ranges capture all pixels that are similar enough to the dominant color

#So defining color boundaries with tolerance. Taking each dominant color, we create a lower and upper bound. Eg [255, 0, 0] with tolerance 50 implies Lower = [205,0,0]     Upper = [255,50,50] (clipped to max 255)
def get_color_bounds(dominant_colors, tolerance=50):
    bounds = []
    for color in dominant_colors:
        lower = np.clip(color - tolerance, 0, 255)
        upper = np.clip(color + tolerance, 0, 255)
        bounds.append((lower, upper))
    return bounds

#creating binary masks
#using our familiar inRange() function, we generate a mask for each color range
#pixels inside range 255 (white)
#pixels outside 0 (black)
color_bounds = get_color_bounds(dominant_colors, tolerance=50)
masks = []
for lower, upper in color_bounds:
    mask = cv2.inRange(img_rgb, lower, upper)
    masks.append(mask)


#Aplly masks to the original image using our favourite bitwiseand to keep only pixels inside the mask
outputs = []
for mask in masks:
    output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    outputs.append(output)

#Show each segmented image separately
for i, output in enumerate(outputs):
    plt.figure()
    plt.imshow(output)
    plt.axis("off")
    plt.title(f"Color Range {i+1}")
    plt.show()

#Analyse the differnce in using the same photo for versions 2 and 3 and make some notes
#There is visible difference
#Try with other image samples I have
#Still need to get deeper into the KMeans algorithm to understand it's nitty-gritty details. I feel semi blind just using it's abstracted methods
#After running a couple of times the color ranges change depending on the order of the dominant color the algorithm displays
#Experiment with HSV for KMeans instead of RGB in next commit
#Maybe wrap this up in a class so that the rest of the team can import this logic, with the Lead's permission that is