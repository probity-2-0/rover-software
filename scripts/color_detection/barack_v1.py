import numpy as np
import cv2 as cv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. HSV ENCODING (CRITICAL FIX)
# -----------------------------
def encode_hsv(hsv):
    h = hsv[:, 0] * 2 * np.pi / 180.0
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    return np.column_stack((np.sin(h), np.cos(h), s, v))


# -----------------------------
# 2. TRAIN DATA
# -----------------------------
X_train = np.array([
    # RED
    [0,255,255],[5,200,200],[175,255,255],[179,200,150],
    # GREEN
    [60,255,255],[75,200,150],[45,150,100],[55,255,120],
    # BLUE
    [120,255,255],[110,200,150],[100,150,100],[125,255,200],
    # YELLOW
    [25,255,255],[30,200,200],[22,180,180],[28,255,150],
    # WHITE
    [0,0,255],[0,10,240],[90,20,255],[30,5,230],
    # NEUTRAL
    [0,0,0],[0,0,50],[90,10,30],[0,0,80]
])

y_train = [
    'Red','Red','Red','Red',
    'Green','Green','Green','Green',
    'Blue','Blue','Blue','Blue',
    'Yellow','Yellow','Yellow','Yellow',
    'White','White','White','White',
    'Neutral','Neutral','Neutral','Neutral'
]

# Encode + scale
X_encoded = encode_hsv(X_train)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train SVM
model = SVC(kernel='rbf', C=12, gamma='scale', probability=True)
model.fit(X_scaled, y_train)


# -----------------------------
# 3. LOAD IMAGE
# -----------------------------
img_path = r"pics\green_balloon.jpg"
original_img = cv.imread(img_path)

if original_img is None:
    print(f"Error: Image not found at {img_path}")
    exit()

display_img = original_img.copy()
hsv_frame = cv.cvtColor(original_img, cv.COLOR_BGR2HSV)


# -----------------------------
# 4. CLICK HANDLER
# -----------------------------
MAX_POINTS = 50
points = []

def identify_color(event, x, y, flags, param):
    global display_img

    if event == cv.EVENT_LBUTTONDOWN:
        # Safe ROI bounds
        y1, y2 = max(0, y-1), min(hsv_frame.shape[0], y+2)
        x1, x2 = max(0, x-1), min(hsv_frame.shape[1], x+2)

        roi = hsv_frame[y1:y2, x1:x2]

        if roi.size == 0:
            hsv_pixel = hsv_frame[y, x].reshape(1, -1)
        else:
            hsv_pixel = roi.reshape(-1, 3).mean(axis=0, keepdims=True)

        # Encode + scale
        encoded = encode_hsv(hsv_pixel)
        scaled = scaler.transform(encoded)

        # Predict (single pass)
        probs = model.predict_proba(scaled)[0]
        idx = np.argmax(probs)

        prediction = model.classes_[idx]
        confidence = probs[idx] * 100

        # Text color based on brightness
        text_color = (0, 0, 255) if hsv_pixel[0, 2] > 200 else (255, 255, 255)

        # Store point
        points.append((x, y, prediction, confidence, text_color))
        if len(points) > MAX_POINTS:
            points.pop(0)

        # Redraw everything (clean overlay)
        display_img = original_img.copy()
        for px, py, name, conf, color in points:
            cv.circle(display_img, (px, py), 5, color, -1)
            cv.putText(display_img, f"{name} {conf:.0f}%",
                       (px + 10, py),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"{prediction} ({confidence:.1f}%) at ({x},{y})")


# -----------------------------
# 5. EXECUTION LOOP
# -----------------------------
cv.namedWindow('Rover Color Inspector')
cv.setMouseCallback('Rover Color Inspector', identify_color)

print("Click image to detect color | 'r' reset | 'q' quit")

while True:
    cv.imshow('Rover Color Inspector', display_img)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        display_img = original_img.copy()
        points.clear()

cv.destroyAllWindows()