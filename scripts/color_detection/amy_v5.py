""" import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

color_dict = {(0,0,0):"black", (255,255,255):"white", (255,0,0):"red", (0,255,0):"green", (0,0,255):"blue", (255,255,0):"yellow", (255,0,255):"magenta", (0,255,255):"cyan"}
color_names = list(color_dict.values())
color_rgb = list(color_dict.keys())
tree = KDTree(color_rgb)

cap = cv2.VideoCapture(0)
clicked = False
box = 20
mouse_x, mouse_y = 0,0
current_text = "Click clck"

def get_color(roi, k=2):
    pixels = roi.reshape((-1,3))
    pixels = np.float32(pixels)
    clstr = KMeans(n_clusters=k, n_init=5)
    clstr.fit(pixels)
    _, counts = np.unique(clstr.labels_, return_counts = True)
    dominant_bgr = clstr.cluster_centers_[np.argmax(counts)].astype(int)
    return dominant_bgr

def mouse_event(event, x, y, flags, param):
    global  mouse_x, mouse_y, current_text, clicked
    mouse_x, mouse_y =x,y
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
           
cv2.namedWindow("new-window")
cv2.setMouseCallback("new-window", mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    x1, y1 = max(0,mouse_x-box//2), max(0, mouse_y-box//2)
    x2, y2 = min(frame.shape[1], mouse_x+box//2), min(frame.shape[0], mouse_y+box//2)
    if clicked:
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            dom_bgr = get_color(roi, k=2)
            b,g,r = int(dom_bgr[0]), int(dom_bgr[1]), int(dom_bgr[2])
            _, i=tree.query((r,g,b))
            color_name = color_names[i]
            current_text = color_name
            print(color_name)        
    clicked = False
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
    cv2.putText(frame, current_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)
    cv2.imshow("new-window", frame)
    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows() """


import cv2
import numpy as np
from sklearn.cluster import KMeans

# We don't need the KDTree or RGB dictionary anymore!

cap = cv2.VideoCapture(0)
clicked = False
box = 20
mouse_x, mouse_y = 0, 0
current_text = "Click somewhere!"

def get_dominant_bgr(roi, k=2):
    # Reshape the image to be a list of pixels
    pixels = roi.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Find the dominant color
    clstr = KMeans(n_clusters=k, n_init=5)
    clstr.fit(pixels)
    _, counts = np.unique(clstr.labels_, return_counts=True)
    
    # Get the BGR values of the most common cluster
    dominant_bgr = clstr.cluster_centers_[np.argmax(counts)].astype(int)
    return dominant_bgr

def get_hsv_color_name(h, s, v):
    # In OpenCV: H is 0-179, S is 0-255, V is 0-255
    
    # Check for Black/White/Gray first using Value and Saturation
    if v < 50:
        return "black"
    if s < 40 and v > 200:
        return "white"
    if s < 50:
        return "gray"

    # If it's not a neutral color, check the Hue
    if h < 10 or h > 165:
        return "red"
    elif 10 <= h < 25:
        return "orange"
    elif 25 <= h < 35:
        return "yellow"
    elif 35 <= h < 85:
        return "green"
    elif 85 <= h < 130:
        return "blue"
    elif 130 <= h < 165:
        return "magenta"
    
    return "unknown"

def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y, current_text, clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True

cv2.namedWindow("new-window")
cv2.setMouseCallback("new-window", mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Define bounding box coordinates safely
    x1, y1 = max(0, mouse_x - box // 2), max(0, mouse_y - box // 2)
    x2, y2 = min(frame.shape[1], mouse_x + box // 2), min(frame.shape[0], mouse_y + box // 2)
    
    if clicked:
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            # 1. Get the dominant BGR color
            dom_bgr = get_dominant_bgr(roi, k=2)
            b, g, r = dom_bgr[0], dom_bgr[1], dom_bgr[2]
            
            # 2. Convert that single BGR pixel to HSV
            # We have to wrap it in a 3D numpy array to use cvtColor
            bgr_pixel = np.uint8([[[b, g, r]]])
            hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_pixel[0][0]
            
            # 3. Get the color name based on the HSV values
            color_name = get_hsv_color_name(h, s, v)
            current_text = color_name
            print(f"Color: {color_name} | HSV: ({h}, {s}, {v})")
            
    clicked = False
    
    # Draw UI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(frame, current_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.imshow("new-window", frame)
    if cv2.waitKey(1) & 0xFF == 27: # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()

