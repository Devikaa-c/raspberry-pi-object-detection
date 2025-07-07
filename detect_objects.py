import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os.path

# --- Constants ---
MODEL_PATH = "efficientdet_lite0.tflite"
LABELS_PATH = "labels.txt"
IMAGE_PATH = "test_image.jpg"
OUTPUT_IMAGE_PATH = "output_image.jpg"
CONFIDENCE_THRESHOLD = 0.3 # Lowered threshold for this model

# --- Load the TFLite model and allocate tensors ---
print("Loading model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("Model Loaded.")

# --- Get model input and output details ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# --- Load the labels ---
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# --- Load and preprocess the image ---
if not os.path.isfile(IMAGE_PATH):
    print(f"FATAL ERROR: {IMAGE_PATH} not found.")
    exit()

print(f"Loading image: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_height, original_width, _ = image.shape
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

# --- Perform detection ---
print("Running inference...")
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
print("Inference Complete.")

# --- Retrieve detection results (CORRECTED BASED ON YOUR DIAGNOSTIC) ---
# Your diagnostic showed the outputs are in a non-standard order.
# We will get them using the correct index from your report.
# output_details[0] (Index 600) is SCORES
# output_details[1] (Index 598) is BOXES
# output_details[2] (Index 601) is NUM_DETECTIONS
# output_details[3] (Index 599) is CLASSES

scores = interpreter.get_tensor(output_details[0]['index'])[0]
boxes = interpreter.get_tensor(output_details[1]['index'])[0]
num_detections = interpreter.get_tensor(output_details[2]['index'])[0]
classes = interpreter.get_tensor(output_details[3]['index'])[0]

# The number of detections is a single float in an array, so we extract and convert to an integer.
num_detections = int(num_detections)

print(f"Found {num_detections} objects. Filtering with threshold > {CONFIDENCE_THRESHOLD}")

# --- Loop through the detected objects ---
for i in range(num_detections):
    if scores[i] > CONFIDENCE_THRESHOLD:
        # Bounding box coordinates are [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = boxes[i]
        
        # Scale coordinates to the original image size
        xmin = int(xmin * original_width)
        xmax = int(xmax * original_width)
        ymin = int(ymin * original_height)
        ymax = int(ymax * original_height)
        
        # Look up the class name
        class_id = int(classes[i])
        class_name = labels[class_id]
        
        # Prepare the label text
        confidence_percent = int(scores[i] * 100)
        label = f'{class_name}: {confidence_percent}%'
        
        # Draw everything on the original image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

# --- Save the final image ---
print(f"Saving final image to: {OUTPUT_IMAGE_PATH}")
cv2.imwrite(OUTPUT_IMAGE_PATH, image)
print("\n--- PROJECT FINISHED SUCCESSFULLY ---")
