import numpy as np
import cv2 as cv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. HSV ENCODING (KEY FIX)
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
    [0,255,255],[5,200,200],[175,255,255],[179,150,100],[2,100,100],
    # ORANGE
    [10,255,255],[15,200,200],[12,255,150],
    # YELLOW
    [25,255,255],[30,200,200],[35,150,150],[28,255,100],
    # GREEN
    [45,255,255],[60,200,200],[75,150,100],[85,255,150],
    # CYAN
    [90,255,255],[95,200,200],
    # BLUE
    [100,255,255],[115,200,200],[125,150,100],[130,255,150],
    # PURPLE
    [140,255,255],[150,200,150],[160,255,200],
    # WHITE
    [0,0,255],[0,10,230],[90,20,255],[45,5,240],
    # NEUTRAL
    [0,0,0],[0,0,50],[0,0,100],[90,5,50],[45,10,80]
])

y_train = (
    ['Red']*5 + ['Orange']*3 + ['Yellow']*4 + ['Green']*4 +
    ['Cyan']*2 + ['Blue']*4 + ['Purple']*3 +
    ['White']*4 + ['Neutral']*5
)

# Encode + scale
X_encoded = encode_hsv(X_train)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train model
model = SVC(kernel='rbf', C=15, gamma='scale', probability=True)
model.fit(X_scaled, y_train)


# -----------------------------
# 3. STATE
# -----------------------------
clicked_points = []
MAX_POINTS = 50


# -----------------------------
# 4. CLICK HANDLER
# -----------------------------
def identify_color(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        hsv_frame = param

        # Safe ROI
        y1, y2 = max(0, y-1), min(hsv_frame.shape[0], y+2)
        x1, x2 = max(0, x-1), min(hsv_frame.shape[1], x+2)

        roi = hsv_frame[y1:y2, x1:x2]
        hsv_pixel = roi.reshape(-1, 3).mean(axis=0, keepdims=True)

        # Encode + scale
        encoded = encode_hsv(hsv_pixel)
        scaled = scaler.transform(encoded)

        # Predict (single pass)
        probs = model.predict_proba(scaled)[0]
        idx = np.argmax(probs)

        prediction = model.classes_[idx]
        confidence = probs[idx] * 100

        # Text color (based on brightness)
        t_color = (0, 0, 255) if hsv_pixel[0, 2] > 200 else (255, 255, 255)

        clicked_points.append((x, y, prediction, confidence, t_color))

        # Limit memory
        if len(clicked_points) > MAX_POINTS:
            clicked_points.pop(0)

        print(f"{prediction} ({confidence:.1f}%) at ({x},{y})")


# -----------------------------
# 5. VIDEO LOOP
# -----------------------------
cap = cv.VideoCapture(0)
cv.namedWindow('Rover Live Feed')

print("Streaming... Click to detect color | 'r' reset | 'q' quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Update callback with current frame
    cv.setMouseCallback('Rover Live Feed', identify_color, hsv_frame)

    # Draw points
    for px, py, name, conf, color in clicked_points:
        cv.circle(frame, (px, py), 5, color, -1)
        cv.putText(frame, f"{name} {conf:.0f}%", (px+10, py),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv.imshow('Rover Live Feed', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        clicked_points.clear()

cap.release()
cv.destroyAllWindows()