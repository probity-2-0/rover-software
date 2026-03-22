import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

color_dict = {(0,0,0):"black", (255,255,255):"white", (255,0,0):"red", (0,255,0):"green", (0,0,255):"blue", (255,255,0):"yellow", (255,0,255):"magenta", (0,255,255):"cyan"}
color_names = list(color_dict.values())
color_rgb = list(color_dict.keys())
tree = KDTree(color_rgb)

img_path = "peeps/rover-software/four_balloons_project/pics/grayson.png"
box = 20
mouse_x, mouse_y = 0,0
img1 = cv2.imread(img_path)
img1 = cv2.resize(img1, (400,400), interpolation=cv2.INTER_AREA)
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
    global  mouse_x, mouse_y, current_text
    mouse_x, mouse_y =x,y
    img2 = img1.copy()
    #cv2.rectangle(img3, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,255,0), 2)
    if event == cv2.EVENT_LBUTTONDOWN:
        top_left_x = max(0, x-box//2)
        top_left_y = max(0, y-box//2)
        bottom_right_x = min(img1.shape[1], x+box//2)
        bottom_right_y = min(img1.shape[0], y+box//2)
        roi = img2[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        if roi.size>0:
            dominant_bgr = get_color(roi, k=2)
            b,g,r = int(dominant_bgr[0]), int(dominant_bgr[1]), int(dominant_bgr[2])
            _, i=tree.query((r,g,b))
            color_name = color_names[i]
            current_text = color_name
            print(color_name)
           
cv2.namedWindow("new-window")
cv2.setMouseCallback("new-window", mouse_event)

while True:
    img3 = img1.copy()
    x1, y1 = max(0,mouse_x-box//2), max(0, mouse_y-box//2)
    x2, y2 = min(img1.shape[1], mouse_x+box//2), min(img1.shape[0], mouse_y+box//2)
    cv2.rectangle(img3, (x1,y1), (x2,y2), (0,255,0), 1)
    cv2.putText(img3, current_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)
    #cv2.putText(img3, current_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2)
    cv2.imshow("new-window", img3)
    if cv2.waitKey(1) & 0xFF==27:
        break

cv2.destroyAllWindows()


