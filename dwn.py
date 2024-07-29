import cv2
import cvzone
import numpy as np
import pytesseract
from ultralytics import YOLO
import torch
import time
from threading import Thread, Event
import math
import os
import pygame
import smtplib
from email.message import EmailMessage

# Initialize pygame mixer
pygame.mixer.init()

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

# Paths to the YOLO configuration and weights files
weapon_cfg_path = 'weapon.cfg'
weapon_weights_path = 'weapon.weights'

# Load YOLO models
weapon_model = cv2.dnn.readNetFromDarknet(weapon_cfg_path, weapon_weights_path)
weapon_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
weapon_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

fire_model = YOLO('fire_model.pt')
number_plate_model = YOLO('license_plate_detector.pt')

# Initialize video capture
stream_url = "http://192.168.1.35:8080/video"
cap = cv2.VideoCapture(stream_url)

# Define class names
fire_classnames = ['fire', 'flame', 'wildfire', 'burn', 'smoke']
weapon_classnames = ['pistol', 'ak47', 'gun', 'longgun', 'knife']
number_plate_classnames = ['number_plate']

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_plate_model.to(device)

def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]

def draw_pred(class_name, confidence, left, top, right, bottom, frame, color=(255, 0, 0)):
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = f'{class_name}: {confidence:.2f}'
    cvzone.putTextRect(frame, label, (left, top), scale=1.5, thickness=2)

def extract_text_from_plate(plate_image):
    # Pre-process the plate image for better OCR results
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary_plate, config='--psm 8')
    return text.strip()

# Text and Color Settings
text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25

# Timing Variables
prev_frame_time = 0
new_frame_time = 0

# Function to read frames asynchronously
def read_frame(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = time.time()
        frame_queue.append((frame, timestamp))
        if len(frame_queue) > 150:
            frame_queue.pop(0)

frame_queue = []
thread = Thread(target=read_frame, args=(cap, frame_queue))
thread.daemon = True
thread.start()

# Siren sound file paths
gun_siren_sound_path = 'gun-siren.mp3'  # Update this path to your gun siren sound file
fire_siren_sound_path = 'fire-siren.mp3'  # Update this path to your fire siren sound file
number_plate_siren_sound_path = 'police-siren.mp3'  # Update this path to your number plate siren sound file

# Flags to control the siren sounds
siren_event = Event()
current_siren = None

# Email configuration
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "akshayghugare@sdlccorp.com"
EMAIL_HOST_PASSWORD = "lhwaoegxcaxvvvla"
RECEIVER_EMAILS = ['gowox96276@mfunza.com', 'akshayghugare0@gmail.com']  # Add multiple email addresses

# Function to play siren sound
def play_siren(siren_sound_path):
    global current_siren
    current_siren = siren_sound_path
    pygame.mixer.music.load(siren_sound_path)
    pygame.mixer.music.play(-1)  # Play the sound on loop

# Function to stop siren sound
def stop_siren():
    pygame.mixer.music.stop()
    global current_siren
    current_siren = None

def send_email(subject, body, attachment_paths):
    msg = EmailMessage()
    msg['From'] = EMAIL_HOST_USER
    msg['To'] = ', '.join(RECEIVER_EMAILS)
    msg['Subject'] = subject
    msg.set_content(body)

    for attachment_path in attachment_paths:
        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
            msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        smtp.send_message(msg)

def process_frame(frame):
    detection_made = False
    detected_type = None

    frame = cv2.resize(frame, (640, 480))

    # Number plate detection
    number_plate_results = number_plate_model(frame, stream=True)
    for info in number_plate_results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if Class < len(number_plate_classnames) and confidence > 50:
                detection_made = True
                detected_type = 'number_plate'
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Extract the number plate region
                plate_image = frame[y1:y2, x1:x2]
                plate_text = extract_text_from_plate(plate_image)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{number_plate_classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                cvzone.putTextRect(frame, f'Text: {plate_text}', [x1 + 8, y1 + 150], scale=1.5, thickness=2)

    # Fire detection
    fire_results = fire_model(frame, stream=True)
    for info in fire_results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                detection_made = True
                detected_type = 'fire'
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{fire_classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)

    # Weapon detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), [0, 0, 0], 1, crop=False)
    weapon_model.setInput(blob)
    outs = weapon_model.forward(get_outputs_names(weapon_model))

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                detection_made = True
                detected_type = 'weapon'
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_pred(weapon_classnames[class_ids[i]], confidences[i], left, top, left + width, top + height, frame)

    return frame, detection_made, detected_type

def save_video_clip(frame_queue, detection_time):
    start_time = detection_time - 5
    end_time = detection_time + 5
    video_clip = []
    for frame, timestamp in frame_queue:
        if start_time <= timestamp <= end_time:
            annotated_frame, _, _ = process_frame(frame)  # Annotate the frame
            video_clip.append(annotated_frame)
    if video_clip:
        filename = f'clips\\detection_{detection_time}.avi'
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        for frame in video_clip:
            out.write(frame)
        out.release()
        return filename
    return None

# Ensure the 'clips' directory exists
if not os.path.exists('clips'):
    os.makedirs('clips')

while True:
    if len(frame_queue) == 0:
        continue

    frame, timestamp = frame_queue[-1]
    processed_frame, detection_made, detected_type = process_frame(frame)

    cv2.imshow('Processed Frame', processed_frame)

    if detection_made:
        detection_time = time.time()
        siren_sound_path = {
            'weapon': gun_siren_sound_path,
            'fire': fire_siren_sound_path,
            'number_plate': number_plate_siren_sound_path
        }.get(detected_type)

        if siren_sound_path and current_siren != siren_sound_path:
            stop_siren()
            play_siren(siren_sound_path)

        screenshot_filename = f'clips\\detection_{detection_time}.jpg'
        cv2.imwrite(screenshot_filename, processed_frame)

        video_clip_filename = save_video_clip(frame_queue, detection_time)

        if video_clip_filename:
            send_email(
                subject=f'Detection Alert: {detected_type.capitalize()} Detected',
                body=f'A {detected_type} was detected. See the attached screenshot and video clip for details.',
                attachment_paths=[screenshot_filename, video_clip_filename]
            )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_siren()
