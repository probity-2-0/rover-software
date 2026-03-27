##add comments
import cv2
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
cv2.destroyAllWindows()


