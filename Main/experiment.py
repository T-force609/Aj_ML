import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class Objectdetection:
    def __init__(self):
        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device name:' ,self.device)

        self.model = self.load_model()

        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame")
                break

            # Convert frame to BGR format (OpenCV expects BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Perform object detection
            results = self.predict(frame)

            # Draw bounding boxes on the frame
            frame = self.plot_bboxes(results, frame)

            # Display the frame with detections
            if cv2 is not None:
                cv2.imshow('Object Detection', results[0].plot())

            # Exit on 'q' press
            if cv2.waitKey(1) == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        

    def load_model(self):
        model = YOLO('/home/aj-segun/Documents/ML/Object-recognition/yolov8s-world.pt')
        model.fuse()

        return model
    
    
    def predict(self, frame):
        results = self.model(frame)

        return results
    

    
    def plot_bboxes(self, results, frame):
        xyxy = []
        confidence =[]
        class_id =[]

        #Extracting result
        for result in results:
            boxes =result.boxes.cpu().numpy()
            print(boxes)

            detections = boxes.xyxy

            #for detection in detections:
                # Draw bounding box and label
                #x_min, y_min, x_max, y_max, conf, class_ids = detection
                #cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                #label = f"{class_ids} ({conf:.2})"  # Format label with confidence score
                #cv2.putText(frame, label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            xyxy.append(boxes.xyxy)
            confidence.append(boxes.conf)
            class_id.append(boxes.cls)

        return frame, confidence, xyxy, class_id
    
    """

    def plot_bboxes(self, frame, detections):
        if not len(detections) == 0:  # Check if detections is empty (list length is 0)
            return frame  # No detections, return original frame

    # Assuming detections are lists containing [x_min, y_min, x_max, y_max, conf, class_id]
        for detection in detections:
            x_min, y_min, x_max, y_max, conf, class_id = detection

        # Draw bounding box and label
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            label = f"{class_id} ({conf:.2f})"
            cv2.putText(frame, label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return frame
        """
    
    def __call__(self):
        pass

if __name__ == "__main__":
    # Replace 0 with your desired webcam index (if multiple cameras)
    capture_index = 0
    detector = Objectdetection()
    detector()



