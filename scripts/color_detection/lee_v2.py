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
#Use mystic force sample image for testing since they are my fvorite rangers and I know it has a good variety of colors to test the detection algorithm on. Plus, it's always fun to work with images of my favorite characters right...Yeah I am kinda lonely ngl 🥲
#path = r"C:\Users\eustace\Downloads\mysticforcesample.webp" #I wonder if .webp is supported by OpenCV, Ik it supports jpg, png and bmp so lemme just try
#path = r"C:\Users\eustace\Downloads\mysticforce.jpeg" #webp is supported btw just trying a new photo, maybe prompt the user to browse for whatever image they want to test on in the future, for now I will just hardcode the path to the image I want to use for testing.
#nkt the ranger images are not showcasing the color detection as well as I expected, maybe I should try a different image with more distinct colors to see if the detection works better. I will try using a colorful landscape photo or something with clear red, green, and blue areas to test the color detection algorithm more effectively.
path = r"C:\Users\eustace\Downloads\colorful_landscape.jpeg" #This image has more distinct colors, so it should work better for testing the color detection algorithm. I will use this image to see if the defined color boundaries can effectively isolate the red, green, and blue regions in the photo.
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
#The first tuple; [17, 15, 100], [50, 56, 200]). 
#I am trying to say that all pixels that have a R >= 100, B >= 15, and G >= 17 along with R <= 200, B <= 56, and G <= 50 will be considered...
boundaries = [ 
    ([17, 15, 100], [50, 56, 200]), #this range captures red colors
    ([86, 31, 4], [220, 88, 50]),#this range isolates the blue colors
    ([25, 146, 190], [62, 174, 250]),#this range captures green colors or cyan?
    ([103, 86, 65], [145, 133, 128]) #this range captures grayish whitish colors
]

#Let us start masking and plotting the images
#So I traverse in the boundaries list, at each iteration I take out the lower limit and the upper limit from the current index of boundaries
#ThenI use the cv2.inRange() function which checks out each pixel in the image and creates a binary mask where pixels that fall within the specified color range are set to 255 (white) and those that do not are set to 0 (black). This allows me to isolate the regions of the image that correspond to the defined color boundaries. The resulting mask is a binary image that highlights the areas of interest based on the color criteria I have defined.
#After that I perform a bitwise AND operation between the original image and the mask to extract the regions of the image that fall within the specified color range. This is done using the cv2.bitwise_and function, which takes the original image and the mask as input and returns an image where only the pixels that correspond to the masked region are retained, while all other pixels are set to black. This allows me to isolate and visualize the areas of the image that match the defined color boundaries.
#I am growing fond of this bitwise AND operation, I saw it in v1 tooo

#Okay spitting ball here; Let us say in the first iteration I wanted pixels with red right, the masking step would have created a mask for the pixels which have red shade and then this bitwise AND operation will allow me to show only red and make all the other color filtered out
#I am definitely reading more about masking, I find original documentation depressing, Youtube maybe?
#Takeaway; masking is essentially a way of telling the computer which parts of the image to pay attention to and which parts to ignore like in OS Signals...Remember?Not the best analogy but works in my head sig_mask and the sig_action functions in the C implementation of Signal handlers

for lower, upper in boundaries:
    lower = np.array(lower, dtype="uint8") #I am converting the lower and upper limits to NumPy arrays of type uint8 (unsigned 8-bit integer) because OpenCV functions typically expect color values to be in this format. This ensures that the color values are correctly interpreted when creating the mask and performing the bitwise AND operation.
    #how does the array know what is lower and what is upper? 
    #Ohhh okay so tuples have lists right and I am unpacking the tuples in the for loop, so in each iteration, lower and upper will be assigned the respective lists from the boundaries tuple. For example, in the first iteration, lower will be assigned [17, 15, 100] and upper will be assigned [50, 56, 200]. This way, I can easily access the lower and upper limits for each color range during the masking process.
    #So does that work kind of indexing into the tuples? Like in the first iteration, lower will be boundaries[0][0] and upper will be boundaries[0][1], and in the second iteration, lower will be boundaries[1][0] and upper will be boundaries[1][1], and so on?
    #tuples...tuples...tuples I should have put more focus on python like I did for C and assembly lol
    #Okay so wait omds how does the for loop know how to iterate without an index? I mean I am used to for loops in C where I have to define an index variable and increment it in each iteration, but in Python, 
    #Can the  for loop  directly iterate over the elements of a list or tuple without needing an explicit index. So in this case, the for loop is iterating over the boundaries list, and in each iteration, it is unpacking the current element (which is a tuple) into the lower and upper variables. Oh right that is why people like python...such conveniences 
    upper = np.array(upper, dtype="uint8")
    
    mask = cv2.inRange(img, lower, upper) #create a binary mask where pixels within the specified color range are set to 255 (white) and those outside the range are set to 0 (black). This allows me to isolate the regions of the image that correspond to the defined color boundaries.
    output = cv2.bitwise_and(img, img, mask=mask) #apply a bitwise AND operation between the original image and the mask to extract the regions of the image that fall within the specified color range. This results in an image where only the pixels that correspond to the masked region are retained, while all other pixels are set to black. This allows me to visualize the areas of the image that match the defined color boundaries. 

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)) #display the output image with the detected color regions highlighted. Again, I am using cv2.cvtColor to convert the color space from BGR to RGB for correct color display in matplotlib.
    plt.axis('off') #turn off the axis labels and ticks for a cleaner display of the image.
    plt.show() #show the image with the detected color regions. This will display the output

#Okay so what if there is no clear red, blue or green  in an image? I will have to opt for like a means to find the colour boundaries dynamically based on the image content. That is where the implementation of K-means clustering(The one I see see Amy yapping about in all her code versions) for color quantization comes in. 
#By using K-means clustering, I can group similar colors together and identify the dominant colors in the image, which can then be used to define the color boundaries for detection. This approach allows for more flexibility and adaptability in color detection, especially in images with a wide range of colors or where the colors are not clearly defined. 
#This is a v3 implementation idea, I will have to read up on K-means clustering and how to apply it for color quantization(Ha Quantization...like in signals...again) in OpenCV. 