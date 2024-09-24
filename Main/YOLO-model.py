
import ultralytics
from dataset.train import annotations
from ultralytics import YOLO



# Create a new YOLO model from scratch
model = YOLO("yolov8.yaml")


dataset1 = annotations


# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=10, imgsz=320, device='cpu', name='object_recognition', pretrained=True)

# Evaluate the model's performance on the validation set
display = model(source=0, show=True, conf=0.6, save=True)
results = model.val()