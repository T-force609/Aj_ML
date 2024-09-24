# dataset/train/_annotations

import csv

class Annotations:
    def __init__(self, filename):
        self.annotations = {}
        self.load_annotations(filename)

    def load_annotations(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # Skip the header row
            next(reader)
            for row in reader:
                image_file = row  # Unpack only the image filename
                annotations = []
                for i in range(4, len(row), 8):  # Skip irrelevant data and jump by 8 for each annotation
                    try:
                        class_name, x_min, y_min, x_max, y_max = row[i:i+5]  # Attempt to unpack 5 values
                        annotations.append({';;'
                            "class": class_name,
                            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                        })
                        self.annotations[image_file] = annotations
                        
                    except ValueError:
                        print(f"Warning: Skipping row with missing data: {row}")  # Handle missing values
                        pass  # Continue to the next row

                    

    def get_annotations(self, image_file):
        return self.annotations.get(image_file, [])  # Return empty list if no annotations found

# Example usage
annotations = Annotations("/home/aj-segun/Documents/ML/Object-recognition/dataset/train/_annotations.csv")  # Replace with your actual CSV filename

image_file = "car_jpeg.rf.45936fb0a7d3f1259f1273bda19d36a6.jpg"
image_data = annotations.get_annotations(image_file)

print(f"Annotations for {image_file}:")
for annotation in image_data:
    print(f"\tClass: {annotation['class']}, Bounding Box: {annotation['bbox']}")