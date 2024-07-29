import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO
import pytesseract

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

# Model Paths
yolo_model_path = 'yolov5su.pt'  # YOLOv5 model path
license_plate_model_path = 'numberplate.pt'  # License plate detection model path
video_path = 'car.mp4'  # Input video path

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
yolo_model = YOLO(yolo_model_path).to(device)
license_plate_model = YOLO(license_plate_model_path).to(device)

# Video Capture and Writer
frame = cv2.VideoCapture(video_path)
frame_width = int(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(frame.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

# Text and Color Settings
text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25

# Timing Variables
prev_frame_time = 0
new_frame_time = 0

# Inference Loop
while True:
    ret, image = frame.read()
    if not ret:
        break

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Model Inference for YOLO
    results = yolo_model(image_rgb)
    detections = results[0].boxes.xyxy.cpu().numpy()

    # Model Inference for License Plate Detection
    results_lp = license_plate_model(image_rgb)
    detections_lp = results_lp[0].boxes.xyxy.cpu().numpy()

    # Combine Results
    combined_results = np.vstack((detections, detections_lp)) if len(detections) > 0 and len(detections_lp) > 0 else (detections if len(detections) > 0 else detections_lp)

    # Process Results
    for i in combined_results:
        p1 = (int(i[0]), int(i[1]))
        p2 = (int(i[2]), int(i[3]))
        text_origin = (int(i[0]), int(i[1]) - 5)

        # Draw bounding boxes
        cv2.rectangle(image, p1, p2, color=color, thickness=2)

        # Extract the license plate region
        plate_img = image[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
        
        # Use Tesseract to extract text
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 7')  # Use Page Segmentation Mode 7
        
        # Draw the extracted text
        cv2.putText(image, plate_text.strip(), text_origin,
                    fontFace=text_font, fontScale=text_font_scale,
                    color=color, thickness=2)

    # Calculate and Display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(image, str(fps), (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Write and Show Frame
    writer.write(image)
    cv2.imshow("image", image)

    # Break on 'q' Key Press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
frame.release()
writer.release()
cv2.destroyAllWindows()
