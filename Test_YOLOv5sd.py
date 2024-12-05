import cv2
import numpy as np
import tensorflow as tf
import random

# Load COCO class names (or any other class names you use)
with open("coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Function to generate a random color for each class
def get_random_color():
    return [random.randint(0, 255) for _ in range(3)]

# Load YOLOv5s TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov5sd.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up the video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define constants
confidence_threshold = 0.5
nms_threshold = 0.4  # Non-maxima suppression threshold

# Create a dictionary to store colors for each class
class_colors = {}

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process the image (resize and normalize)
    input_shape = input_details[0]['shape'][1:3]
    image_resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = np.expand_dims(image_rgb, axis=0).astype(np.float32) / 255.0

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_norm)

    # Run inference
    interpreter.invoke()

    # Get output tensor (shape: [1, 6300, 85])
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # [6300, 85]

    boxes = []
    class_ids = []
    confidences = []

    for detection in output_data:
        # Extract the bounding box, objectness, and class probabilities
        x, y, w, h = detection[0:4]  # Bounding box coordinates
        objectness = detection[4]    # Confidence that the bounding box contains an object
        class_probs = detection[5:]  # Class probabilities (length = 80 for YOLOv5)

        # Find the class with the highest probability
        class_id = np.argmax(class_probs)
        confidence = objectness * class_probs[class_id]

        # Apply confidence threshold
        if confidence > confidence_threshold:
            # Scale bounding boxes back to image size
            h_img, w_img, _ = frame.shape
            x1 = int((x - w / 2) * w_img)
            y1 = int((y - h / 2) * h_img)
            x2 = int((x + w / 2) * w_img)
            y2 = int((y + h / 2) * h_img)

            boxes.append([x1, y1, x2, y2])
            class_ids.append(class_id)
            confidences.append(float(confidence))

    # Apply non-maxima suppression (NMS) to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]

            # Generate or retrieve color for the class
            if class_id not in class_colors:
                class_colors[class_id] = get_random_color()

            color = class_colors[class_id]  # Random color for the class

            # Draw the rectangle with the color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display the class name and confidence score
            label = f"{class_names[class_id]} {confidence*100:.2f}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
