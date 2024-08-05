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
from flask import Flask, jsonify,json,Response, render_template_string, make_response,request

import psycopg2
from psycopg2.extras import RealDictCursor
import datetime
from psycopg2 import OperationalError, DatabaseError,Error 
import boto3
from botocore.exceptions import NoCredentialsError
from flask import Flask, jsonify, Response, render_template_string
from flask_cors import CORS
import cv2
import json
from email.message import EmailMessage
import smtplib
from twilio.rest import Client
from tempfile import NamedTemporaryFile
from moviepy.editor import VideoFileClip
import requests




# AWS S3 Configuration
AWS_ACCESS_KEY = 'test'
AWS_SECRET_KEY = 'test'
S3_BUCKET_NAME = 'testdetection'

# Email configuration
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "akshayghugare@sdlccorp.com"
EMAIL_HOST_PASSWORD = "test"

# Twilio configuration

account_sid = 'test'
auth_token = 'test'
client = Client(account_sid, auth_token)

# Initialize the S3 client
# s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY,)
# Initialize S3 client 
s3_client = boto3.client('s3',
                         aws_access_key_id=AWS_ACCESS_KEY,
                         aws_secret_access_key=AWS_SECRET_KEY)
def upload_to_s3(file_name, S3_BUCKET_NAME, object_name=None):
    object_name = object_name or file_name
    try:
        # Upload the file to S3
        s3_client.upload_file(file_name, S3_BUCKET_NAME, object_name, ExtraArgs={'ACL': 'public-read'})
        
        # Change the ContentType to video/mp4
        s3_client.put_object_acl(ACL='public-read', Bucket=S3_BUCKET_NAME, Key=object_name)
        s3_client.copy_object(Bucket=S3_BUCKET_NAME, CopySource={'Bucket': S3_BUCKET_NAME, 'Key': object_name}, Key=object_name, MetadataDirective='REPLACE', ContentType='video/mp4')
        
        print(f"Upload Successful: {file_name} to {S3_BUCKET_NAME}")
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{object_name}"
        return s3_url 
    except FileNotFoundError:
        print(f"The file was not found: {file_name}")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None

# Example usage
s3_url = upload_to_s3('path_to_your_video.mp4', 'your_s3_bucket_name')
print(s3_url)





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
stream_url = 0
# stream_url="http://192.168.1.78:8080/video"

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
ENCODED_FOLDER="encodedfile"
os.makedirs(save_directory, exist_ok=True)

app = Flask(__name__)

detection_results = {
    'number_plate': [],
    'fire': [],
    'weapon': []
}

# Function to send email when live detection is enabled
def send_email_when_detected(receiver_emails, subject, body, detection_time, detection_type, videofile, imagefile):
    msg = EmailMessage()
    msg['From'] = EMAIL_HOST_USER
    msg['To'] = ', '.join(receiver_emails)
    msg['Subject'] = subject
    print(f"okokok{receiver_emails, subject, body, detection_time, detection_type, videofile, imagefile}")
    html_content = f"""
   <html>
            <body>
                <p>{body}</p>
                <p><strong>Alert Type:</strong> {detection_type}</p>
                <p><strong>Time:</strong> {detection_time}</p>
                <p><strong>Image:</strong></p>
                <img src="{imagefile}" alt="Image" style="max-width:100%; height:auto;">
                <p><strong>Video:</strong></p>
        
                <!-- Video URL as clickable link -->
                <p><a href="{videofile}" target="_blank">Watch the video</a></p>
        
                <!-- Optional: Thumbnail image for better presentation -->
                <a href="{videofile}" target="_blank" style="display: block; text-align: center;">
                  <img src="https://via.placeholder.com/320x240.png?text=Watch+Video" alt="Video Thumbnail" style="max-width:100%; height:auto;">
                </a>
            </body>
        </html>
    """
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            smtp.send_message(msg)
        print("Emails sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to send whats app when live detection is enabled
def send_whatsapp_notification_when_live(phone_numbers, detection_time, detection_type, videofile, imagefile):
    print("Whatsapp notification started")
    messageNotification = "Detected information"
    type = detection_type
    video = videofile
    image = imagefile
    print(f"DATA WhatsApp::{phone_numbers} {type} {videofile} {imagefile}")
    message_sids = []

    for phone_number in phone_numbers:
        formatted_phone_number = f'whatsapp:+91{phone_number}'

        message_body = f"""
        Message: {messageNotification}
        Alert Type: {type}
        Time: {detection_time}
        Image: {image}
        Video: {video}
        """
        try:
            message = client.messages.create(
                from_='whatsapp:+14155238886',
                body=message_body,
                to=formatted_phone_number
            )
            message_sids.append(message.sid)
            print(f"Image message sent to {formatted_phone_number} with SID: {message.sid}")
        except Exception as e:
            print(f"Failed to send message to {formatted_phone_number}: {str(e)}")

    return jsonify({'message_sids': message_sids}), 200

def add_analytics(type, videofile, imagefile):
    print("analytics code------")
    print(f"Type: {type}")
    print(f"Video file: {videofile}")
    print(f"Image file: {imagefile}")
    body = "Detected results"
    subject = "DetectionAlert"
    
    try:
        # Get all user emails
        select_user_email_query = "SELECT email FROM users"
        cursor.execute(select_user_email_query)
        user_email_records = cursor.fetchall()
        
        if not user_email_records:
            print("No users Email found.")
            return "No users Email found."
        
        # Collect emails into a list
        receiver_emails = [user[0] for user in user_email_records]
        print(f"Emails: {receiver_emails}")

        # Get all user Phone number
        select_user_phone_umber_query = "SELECT phone_number FROM users"
        cursor.execute(select_user_phone_umber_query)
        user_phone_number_records = cursor.fetchall()
        
        if not user_phone_number_records:
            print("No users found.")
            return "No users found."
        
        # Collect emails into a list
        receiver_phone_numbers = [user[0] for user in user_phone_number_records]
        print(f"Phone numbers: {receiver_phone_numbers}")
        
        # Create an insert query
        insert_query = """
        INSERT INTO analytics (created_at, images, videos, type)
        VALUES (%s, %s, %s, %s)
        """
        
        # Values to be inserted
        values = (datetime.datetime.now(), imagefile, videofile, type)
        
        # Execute the insert query
        cursor.execute(insert_query, values)
        print(f"Values inserted: {values}")
        
        # Commit the transaction
        connection.commit()

        # Send email to all users when detection is live
        send_email_when_detected(receiver_emails, subject, body, datetime.datetime.now(), type, videofile, imagefile)
        print("Emails sent successfully")
        
        # Send WhatsApp notification to all users when detection is live
        send_whatsapp_notification_when_live(receiver_phone_numbers, datetime.datetime.now(), type, videofile, imagefile)
        print("WhatsApp notifications sent successfully")
       
        
        return "ok"
    
    except Exception as e:
        # Rollback the transaction in case of error
        connection.rollback()
        print(f"Error: {str(e)}")
        return str(e) 

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

# change rencoded video 
def reencode_video(video_url):
    print("Fetching video from URL:", video_url)
    if not video_url:
        return None
    try:
        # Fetch the video from the URL
        response = requests.get(video_url)
        response.raise_for_status()
        
        # Save the video temporarily
        with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(response.content)
            temp_video_path = temp_video.name

        original_filename = os.path.basename(temp_video_path)
        encoded_filename = f'encoded_{os.path.splitext(original_filename)[0]}.mp4'
        encoded_filepath = os.path.join(ENCODED_FOLDER, encoded_filename)

        # Re-encode the video
        clip = VideoFileClip(temp_video_path)
        clip.write_videofile(encoded_filepath, codec='libx264', audio_codec='aac')

        # Clean up the temporary file
        os.unlink(temp_video_path)

        print("Re-encoded video saved as:", encoded_filename)
        return encoded_filepath
    except Exception as e:
        print("Error during re-encoding:", str(e))
        return None



def save_detected_video_and_image(detected_type, detected_frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_filename = os.path.join(save_directory, f'{detected_type}_{timestamp}.mp4')
    image_filename = os.path.join(save_directory, f'{detected_type}_{timestamp}.jpg')

    # Save image with annotation
    cv2.imwrite(image_filename, detected_frame)

    # Save video 10 seconds before and after detection
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (640, 480))

    # Add frames from buffer and 10 seconds after detection
    frames_to_write = list(frame_buffer) + [detected_frame] * 300
    for frame in frames_to_write:
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

        # Re-encode the video and get the new URL
        reencoded_video_path = reencode_video(video_url)
        print(f"Video encoded>>>>{reencoded_video_path}")
        if reencoded_video_path:
            reencoded_video_url = upload_to_s3(reencoded_video_path, S3_BUCKET_NAME, os.path.basename(reencoded_video_path))
            if reencoded_video_url:
                print(f"Re-encoded video uploaded to S3: {reencoded_video_url}")
                add_analytics(detected_type, reencoded_video_url, image_url)

        
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
            fps = 40
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


# start api from this

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Assuming cap is already defined and initialized
# stream_url = "http://192.168.1.78:8080/video"
stream_url = 0
cap = cv2.VideoCapture(stream_url)


def generate_frames():
    print("Current cap:", cap)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        frame = cv2.resize(frame, (640, 480))
        
        # Reset detection flags
        detection_made = False
        detected_type = None

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

        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in indices.flatten():
                box = boxes[i]
                left, top, width, height = box
                draw_pred(weapon_classnames[class_ids[i]], confidences[i], left, top, left + width, top + height, frame)

            detection_results['weapon'].append({
                'class': weapon_classnames[class_ids[i]],
                'confidence': confidences[i],
                'timestamp': time.time()
            })

        # Encode the frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("Failed to encode frame")
            continue
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return 'image:<br><img src="/video_feed"/>'

@app.route('/video_feed')
def video_feed():
    print("yes call to video feed")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# RECEIVER_EMAILS = ['gowox96276@mfunza.com', 'akshayghugare0@gmail.com']  # Add multiple email addresses

def send_email(receiver_emails, subject, body, attachments):
    for attachment in attachments:
        msg = EmailMessage()
        msg['From'] = EMAIL_HOST_USER
        msg['To'] = ', '.join(receiver_emails)
        msg['Subject'] = subject

        html_content = f"""
       <html>
            <body>
                <p>{body}</p>
                <p><strong>Alert Type:</strong> {attachment['type']}</p>
                <p><strong>Time:</strong> {attachment['created_at']}</p>
                <p><strong>Image:</strong></p>
                <img src="{attachment['images']}" alt="Image" style="max-width:100%; height:auto;">
                <p><strong>Video:</strong></p>
        
                <!-- Video URL as clickable link -->
                <p><a href="{attachment['videos']}" target="_blank">Watch the video</a></p>
        
                <!-- Optional: Thumbnail image for better presentation -->
                <a href="{attachment['videos']}" target="_blank" style="display: block; text-align: center;">
                  <img src="https://via.placeholder.com/320x240.png?text=Watch+Video" alt="Video Thumbnail" style="max-width:100%; height:auto;">
                </a>
            </body>
        </html>
        """

        msg.add_alternative(html_content, subtype='html')

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            smtp.send_message(msg)


@app.route('/send-mail', methods=['POST'])
def sendMail():
    try:
        data = request.json
        subject = "Detected analytics"
        body = data.get('body')
        receiver_emails = data.get('receiver_emails')
        mail_type = data.get('type')

        if not receiver_emails or not isinstance(receiver_emails, list):
            return jsonify({"error": "Invalid receiver_emails. It should be a list of email addresses."}), 400

        select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = %s"
        cursor.execute(select_query, (mail_type,))
        records = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        result = [dict(zip(colnames, row)) for row in records]

        if not result:
            return jsonify({"error": "No records found for the given type."}), 404

        send_email(receiver_emails, subject, body, result)
        return jsonify({"message": "Emails sent successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



#Send notification on whats app
@app.route('/send-notification-on-whatsapp', methods=['POST'])
def send_notification():
    data = request.json
    phone_number = data.get('phone_number')
    messageNotification = data.get('message')
    type = data.get('type')

    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = %s"
    cursor.execute(select_query, (type,))
    records = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    result = [dict(zip(colnames, row)) for row in records]

    formatted_phone_number = f'whatsapp:+91{phone_number}'

    message_sids = []
    for record in result:
        message_body = f"""
        Message: {messageNotification}
        Alert Type: {record['type']}
        Time: {record['created_at']}
        Image: {record['images']}
        Video: {record['videos']}
        """
        message = client.messages.create(
            from_='whatsapp:+14155238886',
            body=message_body,
            to=formatted_phone_number
        )
        message_sids.append(message.sid)

    return jsonify({'message_sids': message_sids}), 200


@app.route('/send-perticular-notification-on-whatsapp/<int:id>', methods=['POST'])
def send_one_notification(id):
    data = request.json
    phone_number = data.get('phone_number')
    message_notification = data.get('message')

    # Query to find analytics by ID
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE id = %s"
    cursor.execute(select_query, (id,))
    record = cursor.fetchone()
    
    if not record:
        return jsonify({'error': 'No analytics record found with the given ID'}), 404

    # Extract column names
    colnames = [desc[0] for desc in cursor.description]
    result = dict(zip(colnames, record))

    formatted_phone_number = f'whatsapp:+91{phone_number}'

    message_body = f"""
    Message: {message_notification}
    Alert Type: {result['type']}
    Time: {result['created_at']}
    Image: {result['images']}
    Video: {result['videos']}
    """
    
    # Send WhatsApp message
    message = client.messages.create(
        from_='whatsapp:+14155238886',
        body=message_body,
        to=formatted_phone_number
    )

    return jsonify({'message_sid': message.sid}), 200
    
# POST endpoint to add a new user
@app.route('/add-user', methods=['POST'])
def add_user():
    data = request.get_json()
    name = data.get('name')
    phone_number = data.get('phone_number')
    email = data.get('email')

    # if not name or not phone_number or not email:
    #     return jsonify({"error": "Missing name, phone_number or email"}), 400

    try:
        
        # Create an insert query
        insert_query = """
        INSERT INTO users (name, phone_number, email)
        VALUES (%s, %s, %s)
        RETURNING id, name, phone_number, email
        """
        cursor.execute(insert_query, (name, phone_number, email))
        new_user = cursor.fetchone()
        connection.commit()
        return jsonify(new_user), 201
    except psycopg2.IntegrityError:
        connection.rollback()
        return jsonify({"error": "User with this email already exists"}), 400

# GET endpoint to retrieve all users
@app.route('/get-user', methods=['GET'])
def get_users():
    # Create a select query
    select_query = "SELECT id, name, phone_number, email FROM users"
    cursor.execute(select_query)
    records = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    result = [dict(zip(colnames, row)) for row in records]
    return jsonify(result), 200

@app.route('/get-fire', methods=['GET'])
def getfire():
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = 'fire' ORDER BY created_at DESC"
    cursor.execute(select_query)
    records = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    result = [dict(zip(colnames, row)) for row in records]
    return jsonify(result)

@app.route('/get-numberplate', methods=['GET'])
def getnumberplate():
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = 'number_plate' ORDER BY created_at DESC"
    cursor.execute(select_query)
    records = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    result = [dict(zip(colnames, row)) for row in records]
    return jsonify(result)

@app.route('/get-weapon', methods=['GET'])
def get_weapon():
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type = 'weapon' ORDER BY created_at DESC"
    cursor.execute(select_query)
    records = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    result = [dict(zip(colnames, row)) for row in records]
    return jsonify(result)

@app.route('/get-analytics-by/<int:id>', methods=['GET'])
def get_weapon_by_id(id):
    select_query = "SELECT id, created_at, images, videos, type FROM analytics WHERE type IN ('weapon', 'fire', 'number_plate') AND id = %s"
    cursor.execute(select_query, (id,))
    record = cursor.fetchone()
    if record:
        colnames = [desc[0] for desc in cursor.description]
        result = dict(zip(colnames, record))
        return jsonify(result)
    else:
        return jsonify({"error": "Record not found"}), 404

@app.route('/test', methods=['GET', 'POST'])
def your_endpoint():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)