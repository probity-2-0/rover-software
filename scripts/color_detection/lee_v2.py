# In my v1 I implemented a real-time color detection program using OpenCV in Python. 
# The program captures video from the webcam, converts the color space to HSV, defines color ranges for red, green, and blue, creates masks for each color, applies dilation to enhance the detected regions, and then uses contours to identify and highlight the detected colors in the original video feed. 
# Finally, it displays the results in a full-screen window and allows the user to exit the program by pressing the 'q'
# 
#Before beginning(Oh I am listening to Clairo btw; Bags) it would be cool to used OpenCV for feature extraction like I have done in the past in my trad ML projects, I wonder how different it is 

# For v2 I plan to have 

#So Amy's "Lee" Implementation let's go
#Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt #for visualizing the results, Familiar with this from my ML projects, I can use it to display the original image and the detected color regions side by side for better visualization.
import imutils #for resizing the image, I can use this to resize the input image to a smaller size for faster processing, which is especially useful when working with high-resolution images.
import warnings #for handling warnings, I can use this to suppress any warnings that may arise during the color detection process, such as those related to deprecated functions or potential issues with the input image.
warnings.filterwarnings("ignore") #suppress warnings

#So read the image, print dimensions using the shape attribute(From my v1 I expect 3 dimensions; Length, width, height), display the whole image

path = r"C:\Users\eustace\Downloads\mysticforcesample.webp" #I wonder if .webp is supported by OpenCV, Ik it supports jpg, png and bmp so lemme just try

img = cv2.imread(path)
h, w, d = img.shape #oh right the shape attribute returns a tuple of (height, width, depth) where depth is the number of color channels (3 for RGB images). So I can unpack the shape into h, w, d for easier access to these values later in the code.Had totes forgotten about that.
print(f"Image Dimensions: Height: {h}, Width: {w}, Depth: {d}") #Again, not sure webp format will work

#the .imshow matplot lib function is used to display an image in a window. It takes the image data as input and renders it on the screen. In this case, I am using it to display the original image that I read from the file path. The cv2.cvtColor function is used to convert the color space of the image from BGR (the default color format used by OpenCV) to RGB (the standard color format used by matplotlib).
#This is necessary because OpenCV uses BGR format while matplotlib uses RGB format, so without this conversion, the colors in the displayed image would appear incorrect. By converting the color space, I ensure that the colors are displayed accurately when using matplotlib to show the image.
#remember that cv2.imread reads the image in BGR format by default, so I need to convert it to RGB format before displaying it with matplotlib to ensure that the colors are displayed correctly.
#Wait but why do I need to use the image already read by OpenCV instead of just using the file path directly with matplotlib? I mean sure opencv makes it easier to read and manipulate the image data, but I could just use plt.imread to read the image directly into matplotlib, right? Maybe I can try that later as an alternative approach to see if it works better for my use case. 
#Especially because I am not doing any complex image processing that would require OpenCV's capabilities, so using matplotlib's imread might be sufficient for now...right?

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #display the original image, I am using cv2.cvtColor to convert the color space from BGR to RGB for correct color display in matplotlib. My dumbass thought bgr and rgb are the same.
plt.axis('off') # turns off the axis labels and ticks, I do not need any distractions when displaying images my Self diagnosed ADHD will go berserk 🤡

plt.show()#show lol

#Defining boundaries
#Okay...why do I need boundaries for the BGR color space 
#Each tuple takes in 2 lists First one defining the lower limit and the second one defining the upper limit
#Since I am having a hard time with Tuples I will knowingly make my life more difficult
#The first tuple; [17, 15, 100], [50, 56, 200]). This tuple means
boundaries = [ 
    ([17, 15, 100], [50, 56, 200])
    ([86, 31, 4], [220, 88, 50])
    ([25, 146, 190], [62, 174, 250])
    ([103, 86, 65], [145, 133, 128]) 
]
