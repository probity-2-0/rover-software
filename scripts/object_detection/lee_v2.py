from ultralytics import YOLO

# Load the pre-trained YOLOv8 nano model (fastest for real-time)
model = YOLO('yolov8n.pt')

# Run inference on your IP camera stream
# 'show=True' tells YOLO to pop up a window with detections
model.predict(source="http://192.168.137.14:8080/video", show=True, conf=0.5)
