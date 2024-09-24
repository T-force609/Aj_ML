import cv2
import torch
from time import time
import numpy
from ultralytics import YOLO 

# Load the YOLO model (replace 'best.pt' with your model path)
model = YOLO('yolov8s-world.pt')  # Load pre-trained or custom weights

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Loop for video capture and object detection
while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from webcam")
        break

    # Make predictions using the YOLO model
    results = model(frame, stream=True)

    # Loop through detected objects (if any)
    def plot_bboxes(self, results, frame):
        xyxy = []
        confidence = []
        class_id = []

        for data in results.xyxy[0]:
        # Extract confidence score (optional)
            #confidence = float(data['confidence'])
            boxes = data.boxes.cpu().numpy()

            xyxys = boxes.xyxy
            print(xyxys)

        # Set a minimum confidencqe threshold (adjust as needed)
        if xyxy in xyxys:
            # Extract bounding box coordinates
            xmin, ymin, xmax, ymax = int(data['xmin']), int(data['ymin']), int(data['xmax']), int(data['ymax'])

            # Extract object class label
            class_name = data['name']

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green for detected objects
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)  # Green text
            
        return frame

    # Display the resulting frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture resources
cap.release()
cv2.destroyAllWindows()