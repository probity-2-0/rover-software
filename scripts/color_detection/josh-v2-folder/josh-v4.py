import cv2
import numpy as np
import os


# the 80 object classes the YOLO model was trained on, in order
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush", "balloon"
]

# minimum confidence to accept a detection
CONFIDENCE_THRESHOLD = 0.3

# how aggressively to remove duplicate boxes (0-1, lower = more aggressive)
NMS_THRESHOLD = 0.4

# the size YOLO expects images to be
MODEL_INPUT_SIZE = (640, 640)


def get_color_name(crop):
    # convert the cropped region to HSV so hue is easy to measure
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])

    # low saturation means the color is washed out — call it white or gray
    if avg_sat < 50:
        avg_val = np.mean(hsv[:, :, 2])
        return "white" if avg_val > 127 else "gray"

    # red wraps around both ends of the hue scale
    if avg_hue < 15 or avg_hue > 155:
        return "red"
    elif avg_hue < 30:
        return "orange"
    elif avg_hue < 45:
        return "yellow"
    elif avg_hue < 85:
        return "green"
    elif avg_hue < 130:
        return "blue"
    else:
        return "purple"


def draw_box(image, x1, y1, x2, y2, color_name):
    # draw the bounding box around the object
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # draw a filled rectangle as the label background
    cv2.rectangle(image, (x1, y1 - 25), (x1 + 120, y1), (0, 255, 0), -1)

    # write the color name on the label background
    cv2.putText(image, color_name, (x1 + 4, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def parse_detections(raw_output, img_width, img_height):
    # the model output is shaped (1, 84, 8400) - squeeze it to (84, 8400)
    # 84 = 4 box coords + 80 class scores, 8400 = candidate detections
    output = raw_output[0].squeeze()

    boxes = []
    confidences = []
    class_ids = []

    for i in range(output.shape[1]):
        scores = output[4:, i]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # skip weak detections
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # YOLO gives center x, center y, width, height scaled to 640x640
        # convert back to original image pixel coordinates
        cx = output[0, i] * img_width / MODEL_INPUT_SIZE[0]
        cy = output[1, i] * img_height / MODEL_INPUT_SIZE[1]
        w  = output[2, i] * img_width / MODEL_INPUT_SIZE[0]
        h  = output[3, i] * img_height / MODEL_INPUT_SIZE[1]

        # convert from center format to corner format
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)

        boxes.append([x1, y1, int(w), int(h)])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # remove overlapping duplicate boxes, keep only the best one per object
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    return boxes, confidences, class_ids, indices


def process_image(image_path):
    # load the ONNX model
    model = cv2.dnn.readNetFromONNX("yolov8n.onnx")

    # load the image from disk
    image = cv2.imread(image_path)

    # check the image actually loaded before doing anything else
    if image is None:
        print(f"Error: could not load image at '{image_path}'")
        print(f"Current working directory: {os.getcwd()}")
        return

    img_height, img_width = image.shape[:2]
    print(f"Image loaded: {img_width}x{img_height} pixels")

    # prepare the image into the format YOLO expects
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, MODEL_INPUT_SIZE, swapRB=True)

    # pass the image through the model
    model.setInput(blob)
    raw_output = model.forward()

    # parse the raw numbers into usable detection data
    boxes, confidences, class_ids, indices = parse_detections(raw_output, img_width, img_height)

    if len(indices) == 0:
        print("No objects detected — try lowering CONFIDENCE_THRESHOLD")
        return

    # loop over the detections that survived NMS
    for i in indices:
        x1, y1, w, h = boxes[i]
        x2 = x1 + w
        y2 = y1 + h

        # keep box coordinates within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)

        # cut out just the pixels inside this box
        crop = image[y1:y2, x1:x2]

        # skip if the crop is somehow empty
        if crop.size == 0:
            continue

        color_name = get_color_name(crop)
        draw_box(image, x1, y1, x2, y2, color_name)

        print(f"Found: {CLASS_NAMES[class_ids[i]]} — color: {color_name} — confidence: {confidences[i]:.2f}")

    # save the annotated image as a png to avoid format issues
    output_path = "output_" + os.path.splitext(image_path)[0] + ".png"
    cv2.imwrite(output_path, image)
    print(f"\nSaved to {output_path}")

    # copy to windows desktop for easy viewing
    desktop = f"/mnt/c/Users/{os.environ.get('USER', 'Admin')}/Desktop/"
    if os.path.exists(desktop):
        cv2.imwrite(desktop + "output_result.png", image)
        print(f"Also copied to Windows Desktop as output_result.png")


def main():
    # change this to your test image filename
    process_image("red_balloon.jpg")


if __name__ == "__main__":
    main()
