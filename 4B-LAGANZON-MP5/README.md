# CSST106 - Perception and Computer Vision: Machine Problem 5


### Jonathan Q. Laganzon
### BSCS-4B

## Overview
This project demonstrates how to set up and use YOLOv3 (You Only Look Once version 3), a real-time object detection system using python.


### Model Loading
In this section, we load the YOLOv3 model, which involves three key files:
1. **Configuration file (`yolov3.cfg`)** - Defines the structure of the YOLOv3 model.
2. **Weights file (`yolov3.weights`)** - Contains the trained parameters of the model, enabling it to recognize objects.
3. **Class names file (`coco.names`)** - Lists the names of all object classes the model can detect, like "person," "car," or "dog."

These files must be properly referenced for YOLOv3 to work. If you're using Google Drive to store these files, this section also explains how to mount your Google Drive for easy file access.

---

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
```

### Object Detection
Object detection is the core functionality of YOLOv3. In this part:
1. **Preparing the image** - The input image is processed and resized to fit the model’s input size requirements.
2. **Running detection** - YOLOv3 analyzes the image to detect objects, using its network layers and pre-trained weights.
3. **Output processing** - The model returns details about each detected object, including:
   - **Bounding boxes** - The rectangular areas around each detected object.
   - **Class labels** - The names of the objects (like "cat," "car").
   - **Confidence scores** - The probability that the detection is accurate.

This process provides the coordinates and information needed to draw boxes around detected objects and label them with their class names and confidence scores.

---
```python
def detect_objects(image):
    height, width = image.shape[:2]

    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

    # Run inference
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes
```

### Visualization
The visualization step makes the detection results easy to interpret by:
1. **Drawing bounding boxes** around each detected object.
2. **Adding labels** with class names and confidence scores beside each object.

Visualization helps confirm if the model is correctly identifying objects. The bounding boxes and labels clearly indicate where and what objects are found in the image, making it easy to evaluate the model's performance at a glance.

---


```python
def draw_boxes(image, boxes, confidences, class_ids, indexes):
    for i in indexes.flatten():
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image
image_path = "/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found. Check the image path.")

boxes, confidences, class_ids, indexes = detect_objects(image)
result_image = draw_boxes(image, boxes, confidences, class_ids, indexes)

# Convert BGR to RGB for displaying with matplotlib
result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

# Show the output using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(result_image)
plt.axis('off')
plt.show()
```
![mp5-viz](https://github.com/user-attachments/assets/9aa90000-f175-47e7-bdd1-55478db572fe)



### Testing
Testing allows you to evaluate YOLOv3’s detection abilities on a set of images. In this section:
1. **Load test images** - Select images on which you want to test the model.
2. **Run the object detection function** - Each image goes through the detection process.
3. **View results** - The detected objects are displayed with bounding boxes and labels, and detection performance (like time taken) is printed.

Testing provides insights into how well the model performs across different images, showing its accuracy, speed, and capability in varied scenarios. 

---

```python
import time
import cv2
import matplotlib.pyplot as plt

# List of image paths for testing
image_paths = [
    "/content/drive/MyDrive/2x2/dogs.jpg",
    "/content/drive/MyDrive/2x2/cars.jpg",
    "/content/drive/MyDrive/2x2/motor.jpg"
]

# Function to test and evaluate YOLO on multiple images
def test_yolo_on_images(image_paths):
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image {idx + 1} not found. Check the image path: {image_path}")
            continue

        # Start timer to measure detection time
        start_time = time.time()

        # Run detection
        boxes, confidences, class_ids, indexes = detect_objects(image)

        # End timer and calculate elapsed time
        elapsed_time = time.time() - start_time

        # Draw boxes on the image
        result_image = draw_boxes(image, boxes, confidences, class_ids, indexes)

        # Convert BGR to RGB for displaying with matplotlib
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Show the output using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image)
        plt.axis('off')
        plt.title(f"Image {idx + 1} - Detection Time: {elapsed_time:.2f} seconds")
        plt.show()

        # Print performance details for each image
        print(f"Image {idx + 1} - Detection Time: {elapsed_time:.2f} seconds")
        print(f"Detected Objects: {len(indexes)}")
        print(f"Detection Details: Boxes - {len(boxes)}, Confidences - {len(confidences)}\n")

# Call the test function
test_yolo_on_images(image_paths)
```


![mp5-test1](https://github.com/user-attachments/assets/5b1ead6c-f56a-4432-9a3c-91f76dc7e205)

![mp5-test2](https://github.com/user-attachments/assets/85d88061-efda-42f1-86f3-78f7c04192d5)

![mp5-test3](https://github.com/user-attachments/assets/e82a0c16-9a25-4961-bc06-a7fd26a6c8c6)


## Performance Analysis

In this section, we analyze the model’s performance by looking at detection times, accuracy, and the number of objects detected in each test image. This analysis helps us understand YOLOv3’s strengths in real-time object detection and how its single-pass detection mechanism contributes to its speed and effectiveness.

---

### Key Observations:
1. **Speed and Detection Time**: YOLOv3 performs a single pass through the image to detect objects, which contributes to its speed. The times recorded for each image show the model's capability for real-time applications, where detection times under 1.5 seconds are generally considered responsive.
2. **Accuracy**: YOLOv3's accuracy is evident from its confidence scores and correct bounding of detected objects. The high confidence scores show that YOLO can accurately detect and identify objects even in complex or busy scenes.

---

### Test Results and Analysis for Each Image

---

### Image 1
- **Detection Time**: 0.93 seconds
- **Objects Detected**: 5
- **Detection Details**:
  - Bounding Boxes: 13
  - Confidence Scores: 13

**Analysis**:  
In this image, the model successfully detected multiple instances of the "dog" class with high confidence, ranging from 0.71 to 1.00. The bounding boxes were precise and consistent across similar objects.

**Observation**:  
YOLOv3’s ability to identify multiple similar objects (e.g., puppies) in a single image demonstrates its reliability and accuracy. The high confidence scores reflect the model’s effectiveness in distinguishing repeated instances of the same object within the frame.

---

### Image 2
- **Detection Time**: 1.03 seconds
- **Objects Detected**: 5
- **Detection Details**:
  - Bounding Boxes: 19
  - Confidence Scores: 19

**Analysis**:  
In a busier background with several cars, YOLOv3 detected multiple objects with varying confidence levels. Despite the scene's complexity, the model was able to distinguish each car and provided accurate bounding boxes.

**Observation**:  
YOLOv3 performs well in complex scenes, maintaining high accuracy and fast detection. Its single-pass architecture enables it to quickly identify and locate multiple instances of similar objects (like cars), showing its suitability for real-world environments with numerous, repetitive items.

---

### Image 3
- **Detection Time**: 1.35 seconds
- **Objects Detected**: 2
- **Detection Details**:
  - Bounding Boxes: 8
  - Confidence Scores: 8

**Analysis**:  
This image included different types of objects—a "person" and a "motorbike." YOLOv3 correctly identified both with high confidence, scoring up to 1.00 for "person" and 0.99 for "motorbike."

**Observation**:  
YOLOv3 effectively handles diverse scenes with different object classes. The high accuracy in detecting distinct object types highlights the model's flexibility and adaptability to a variety of objects in a single frame.

---

The YOLOv3 model demonstrates a balance of speed and accuracy, thanks to its single-pass detection approach. This enables YOLO to quickly analyze an image and deliver high confidence in detection, even in complex and diverse scenes. YOLO’s structure is optimized for real-time applications, making it a powerful tool for scenarios requiring fast, accurate object detection across varied environments.
---

###Summary
This project offers a comprehensive workflow to load, run, and analyze an object detection model (YOLOv3) using Python. It includes sections for visualization, testing, and performance evaluation to ensure a complete understanding of the model's capabilities
