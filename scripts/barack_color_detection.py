#imports
import numpy as np
import cv2 as cv

# webcam = cv.VideoCapture(0)

# while (1):
#     _,img=webcam.read()    
# load image
img=cv.imread('../four_balloons_project/pics/blue-balloon.jpg')
#display
cv.imshow('Blue',img)
#Convert to HSV
hsvFrame=cv.cvtColor(img,cv.COLOR_BGR2HSV)
#Colors
    #Red
red_lower = np.array([136,87,111],np.uint8)    
red_upper = np.array([180,255,255],np.uint8)    
red_mask = cv.inRange(hsvFrame,red_lower,red_upper)
    #Green   
green_lower = np.array([35,52,72],np.uint8)    
green_upper = np.array([85,255,255],np.uint8)    
green_mask = cv.inRange(hsvFrame,green_lower,green_upper)  
    #Blue
blue_lower = np.array([94,80,2],np.uint8)    
blue_upper = np.array([120,255,255],np.uint8)    
blue_mask = cv.inRange(hsvFrame,blue_lower,blue_upper) 
    #Yellow  
yellow_lower = np.array([20,80,80],np.uint8)    
yellow_upper = np.array([30,255,255],np.uint8)    
yellow_mask = cv.inRange(hsvFrame,yellow_lower,yellow_upper)  
#Masks
kernel=np.ones((5,5),"uint8")
    #red
red_mask=cv.dilate(red_mask,kernel)
res_red=cv.bitwise_and(img,img,mask=red_mask) 
    #green
green_mask=cv.dilate(green_mask,kernel)
res_green=cv.bitwise_and(img,img,mask=green_mask) 
    #blue
blue_mask=cv.dilate(blue_mask,kernel)
res_blue=cv.bitwise_and(img,img,mask=blue_mask)
    #yellow
yellow_mask=cv.dilate(yellow_mask,kernel)
res_yellow=cv.bitwise_and(img,img,mask=yellow_mask)
#Contours
    #red
contours,hierarchy=cv.findContours(red_mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)    
for pic,contour in enumerate(contours):
    area=cv.contourArea(contour)
    if(area>300):
        x,y,w,h=cv.boundingRect(contour)
        img=cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv.putText(img,"Red Color",(x,y),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)    

    #green
contours,hierarchy=cv.findContours(green_mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)    
for pic,contour in enumerate(contours):
    area=cv.contourArea(contour)
    if(area>300):
        x,y,w,h=cv.boundingRect(contour)
        img=cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(img,"Green Color",(x,y),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)    

    #blue
contours,hierarchy=cv.findContours(blue_mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)    
for pic,contour in enumerate(contours):
    area=cv.contourArea(contour)
    if(area>300):
        x,y,w,h=cv.boundingRect(contour)
        img=cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv.putText(img,"Blue Color",(x,y),cv.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),2)    

    #yellow
contours,hierarchy=cv.findContours(yellow_mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)    
for pic,contour in enumerate(contours):
    area=cv.contourArea(contour)
    if(area>300):
        x,y,w,h=cv.boundingRect(contour)
        img=cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cv.putText(img,"Yellow Color",(x,y),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)    

#Aplication
cv.imshow('Color Detection',img)    
if cv.waitKey(0) & 0xFF == ord('q'):
    # webcam.release()
    cv.destroyAllWindows()
    # break