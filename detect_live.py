import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# --- Constants ---
MODEL_PATH = "efficientdet_lite0.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.5
# Use a lower resolution for higher FPS on the Raspberry Pi
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Function to draw bounding boxes and labels ---
def draw_detection(frame, box, class_name, score):
    """Draws a single detection on the frame."""
    ymin, xmin, ymax, xmax = box
    frame_height, frame_width, _ = frame.shape
    xmin = int(xmin * frame_width)
    xmax = int(xmax * frame_width)
    ymin = int(ymin * frame_height)
    ymax = int(ymax * frame_height)
    label = f'{class_name}: {int(score * 100)}%'
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
    cv2.putText(frame, label, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 255, 0), 2)

# --- Main execution ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = input_details[0]['shape']

with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started. Press 'q' to quit.")
print("A window will open on your monitor showing the live feed.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_width, input_height))
    input_data = np.expand_dims(image_resized, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])
    classes = interpreter.get_tensor(output_details[3]['index'])[0]

    for i in range(num_detections):
        if scores[i] > CONFIDENCE_THRESHOLD:
            draw_detection(frame, boxes[i], labels[int(classes[i])], scores[i])

    cv2.imshow('Real-Time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Application closed.")
