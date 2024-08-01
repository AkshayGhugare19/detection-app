import cv2
import cvzone
import numpy as np
import pytesseract
from ultralytics import YOLO
import torch
import time
from threading import Thread
import math
from collections import deque
import os
from flask import Flask, jsonify,json,Response, render_template_string
import psycopg2
from psycopg2 import OperationalError, DatabaseError,Error 
import boto3
from datetime import datetime

from botocore.exceptions import NoCredentialsError



# AWS S3 Configuration
AWS_ACCESS_KEY = 'test'
AWS_SECRET_KEY = 'test'
S3_BUCKET_NAME = 'test'

# Initialize the S3 client
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

def upload_to_s3(file_name, bucket, object_name=None):
    try:
        s3_client.upload_file(file_name, bucket, object_name or file_name)
        print(f"Upload Successful: {file_name} to {bucket}")
        s3_url = f"https://{bucket}.s3.amazonaws.com/{object_name or file_name}"
        return s3_url
    except FileNotFoundError:
        print(f"The file was not found: {file_name}")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None





try:
    # Establish a connection to the PostgreSQL database
    connection = psycopg2.connect(
        user="postgres",
        password="root",
        host="localhost",
        port="5432",
        database="detection"
    )

    # Create a cursor object using the connection
    cursor = connection.cursor()

    # Execute a sample query
    cursor.execute("SELECT version();")
    
    # Fetch one result
    record = cursor.fetchone()
    print(f"You are connected to - {record}\n")

except (Exception, Error) as error:
    print(f"Error while connecting to PostgreSQL: {error}")



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
stream_url = "https://192.168.1.5:8080/video"
cap = cv2.VideoCapture(stream_url)

# Define class names
fire_classnames = ['fire', 'flame', 'wildfire', 'burn', 'smoke']
weapon_classnames = ['pistol', 'ak47', 'gun', 'longgun', 'knife']
number_plate_classnames = ['number_plate']

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_plate_model.to(device)

# Create directory for saving detected videos and images
save_directory = 'detectedfile'
os.makedirs(save_directory, exist_ok=True)

app = Flask(__name__)

detection_results = {
    'number_plate': [],
    'fire': [],
    'weapon': []
}

def add_analytics(type,videofile,imagefile):
    print("analytics code------")
    print(type)
    print(videofile)
    print(imagefile)

     # Create an insert query
    insert_query = """
    INSERT INTO analytics (created_at, images, videos, type)
    VALUES (%s, %s, %s,%s)
    """
    
    # Values to be inserted
    values = (datetime.now(), imagefile, videofile,type)

    # Execute the insert query
    cursor.execute(insert_query, values)

    # Commit the transaction
    connection.commit()

    return "ok"


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

# Buffer to store frames
frame_buffer = deque(maxlen=300)  # 10 seconds at 30 fps

# Function to read frames asynchronously
def read_frame(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.append(frame)
        frame_buffer.append(frame)
        if len(frame_queue) > 1:
            frame_queue.pop(0)

frame_queue = []
thread = Thread(target=read_frame, args=(cap, frame_queue))
thread.daemon = True
thread.start()

def save_detected_video_and_image(detected_type, detected_frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_filename = os.path.join(save_directory, f'{detected_type}_{timestamp}.avi')
    image_filename = os.path.join(save_directory, f'{detected_type}_{timestamp}.jpg')

    # Save image with annotation
    cv2.imwrite(image_filename, detected_frame)

    # Save video 10 seconds before and after detection
    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    for frame in list(frame_buffer) + [detected_frame] * 300:  # 10 seconds after
        video_writer.write(frame)
    video_writer.release()


      # Upload image to S3 and get the URL
    image_url = upload_to_s3(image_filename, S3_BUCKET_NAME, os.path.basename(image_filename))
    if image_url:
        print(f"Image uploaded to S3: {image_url}")

    # Upload video to S3 and get the URL
    video_url = upload_to_s3(video_filename, S3_BUCKET_NAME, os.path.basename(video_filename))
    if video_url:
        print(f"Video uploaded to S3: {video_url}")
        add_analytics(detected_type,image_url,video_url)

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

                detection_results['number_plate'].append({
                    'text': plate_text,
                    'confidence': confidence,
                    'timestamp': time.time()
                })

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

                detection_results['fire'].append({
                    'class': fire_classnames[Class],
                    'confidence': confidence,
                    'timestamp': time.time()
                })

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
        left, top, width, height = box
        draw_pred(weapon_classnames[class_ids[i]], confidences[i], left, top, left + width, top + height, frame)

        detection_results['weapon'].append({
            'class': weapon_classnames[class_ids[i]],
            'confidence': confidences[i],
            'timestamp': time.time()
        })

    return frame, detection_made, detected_type

def detection_loop():
    while True:

        if frame_queue:
            prev_frame_time = 0 
            frame = frame_queue[0]
            processed_frame, detection_made, detected_type = process_frame(frame)

            # Calculate and Display FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            cv2.putText(processed_frame, f'FPS: {fps}', (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Save detected video and image
            if detection_made:
                save_detected_video_and_image(detected_type, processed_frame)

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

detection_thread = Thread(target=detection_loop)
detection_thread.daemon = True
detection_thread.start()

@app.route('/api/detections', methods=['GET'])
def get_detections():
    return jsonify(detection_results)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Stream</title>
    </head>
    <body>
        <h1>Video Stream</h1>
        <img src="{{ url_for('video_feed') }}" style="width: 100%; "/>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get-fire')

def getfire():
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = 'fire'"
    
    # Execute the select query
    cursor.execute(select_query)
    
    # Fetch all rows from the executed query
    records = cursor.fetchall()
    
    # Column names
    colnames = [desc[0] for desc in cursor.description]
    
    # Convert query results to a list of dictionaries
    result = [dict(zip(colnames, row)) for row in records]
    
    # Convert the result to JSON format
    result_json = json.dumps(result, default=str)
    
    print("Query results in JSON format:")
    print(result_json)

   
    

    return jsonify(result_json)

@app.route('/get-numberplate')

def getnumberplate():
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = 'number_plate'"
    
    # Execute the select query
    cursor.execute(select_query)
    
    # Fetch all rows from the executed query
    records = cursor.fetchall()
    
    # Column names
    colnames = [desc[0] for desc in cursor.description]
    
    # Convert query results to a list of dictionaries
    result = [dict(zip(colnames, row)) for row in records]
    
    # Convert the result to JSON format
    result_json = json.dumps(result, default=str)
    
    print("Query results in JSON format:")
    print(result_json)

   
    

    return jsonify(result_json)


@app.route('/get-weapon')

def get_wepon():
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = 'weapon'"
    
    # Execute the select query
    cursor.execute(select_query)
    
    # Fetch all rows from the executed query
    records = cursor.fetchall()
    
    # Column names
    colnames = [desc[0] for desc in cursor.description]
    
    # Convert query results to a list of dictionaries
    result = [dict(zip(colnames, row)) for row in records]
    
    # Convert the result to JSON format
    result_json = json.dumps(result, default=str)
    
    print("Query results in JSON format:")
    print(result_json)

   
    

    return jsonify(result_json)


if __name__ == '__main__':
    app.run(debug=True)