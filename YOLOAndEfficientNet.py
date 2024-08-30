import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# Load YOLO model
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load the class labels YOLO was trained on
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the input image
image = cv2.imread("plant_image.jpg")
height, width, channels = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo_net.setInput(blob)
outputs = yolo_net.forward(output_layers)

# Analyze the detections from YOLO
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Perform non-maxima suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and crop detected plants
detected_plants = []
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    label = str(classes[class_ids[i]])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Crop the detected plant from the image
    cropped_img = image[y:y+h, x:x+w]
    detected_plants.append(cropped_img)

# Display the image with bounding boxes
cv2.imshow("Detected Plants", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load EfficientNetB0 model pre-trained on ImageNet
efficient_net = EfficientNetB0(weights="imagenet")

# Function to predict disease using EfficientNet
def predict_disease(img):
    # Preprocess the image
    img_array = cv2.resize(img, (224, 224))
    img_array = img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the disease
    preds = efficient_net.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return decoded_preds

# Analyze each detected plant with EfficientNet for disease recognition
for idx, plant_img in enumerate(detected_plants):
    predictions = predict_disease(plant_img)
    print(f"Predictions for detected plant {idx + 1}:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{i + 1}: {label} ({score:.2f})")
