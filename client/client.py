"""Main surveillance loop handling video capture and detection tasks.

This script connects to a camera, detects humans using an SSD MobileNet model,
extracts faces, and uploads cropped images to a remote Flask server.
"""

import json
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import time
import cv2
import queue
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import requests
import RPi.GPIO as GPIO


def send_image(image_path, server_url):
    """Send an image file to the prediction server via HTTP."""
    print(image_path.split("/")[-1])

    files = {
        "image": (
            f"{image_path.split('/')[-1]}",
            open(image_path, "rb"),
            "image/jpeg",
        )
    }
    response = requests.post(server_url, files=files)
    print(f"{response} for {image_path}")
    if response.status_code == 200:
        print("Image successfully sent!")
    else:
        print("Failed to send image. Status code:", json.loads(response.text))


def detect_faces(image, frame_count, human_count):
    """Detect faces in ``image`` and send them to the server."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30)
    )

    print(len(faces))

    for i, (x, y, w, h) in enumerate(faces):
        print("Face detected")
        face = image[y : y + h, x : x + w]
        image_path = os.path.join("faces", f"faces_{frame_count}_{human_count}_{i}.jpg")
        cv2.imwrite(image_path, face)
        server_url = "http://192.168.8.157:5001/upload"
        send_image(image_path, server_url)

    return faces


def detect_human(image_np, frame_count):
    """Detect humans in a frame and upload cropped regions."""
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)
    results = model(image_tensor)

    detection_classes = results["detection_classes"][0].numpy().astype(np.int32)
    detection_scores = results["detection_scores"][0].numpy()
    detection_boxes = results["detection_boxes"][0].numpy()

    selected_indices = tf.image.non_max_suppression(
        detection_boxes,
        detection_scores,
        max_output_size=10,
        iou_threshold=0.5,
        score_threshold=0.5,
    )

    selected_boxes = tf.gather(detection_boxes, selected_indices).numpy()
    selected_scores = tf.gather(detection_scores, selected_indices).numpy()
    selected_classes = tf.gather(detection_classes, selected_indices).numpy()

    count = 1
    for i in range(len(selected_classes)):
        if selected_classes[i] == 1:
            y1, x1, y2, x2 = selected_boxes[i]
            y1, x1, y2, x2 = (
                int(y1 * image_np.shape[0]),
                int(x1 * image_np.shape[1]),
                int(y2 * image_np.shape[0]),
                int(x2 * image_np.shape[1]),
            )
            cropped_image_np = image_np[y1:y2, x1:x2]
            # detect_faces(cropped_image_np, frame_count, count)
            cropped_image = Image.fromarray(cropped_image_np)
            cropped_image_path = f"images/cropped_image{frame_count}_{count}.jpg"
            cropped_image.save(cropped_image_path)
            server_url = "http://192.168.8.157:5001/upload"
            send_image(cropped_image_path, server_url)
            count += 1


def producer(cap, frame_queue, start_time, duration=30, every_n_frame=5):
    """Read frames from ``cap`` and queue them for processing."""
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or time.time() - start_time > duration:
            break
        if frame_count % every_n_frame == 0:
            frame_queue.put((frame, frame_count))
        frame_count += 1
    frame_queue.put(None)  # Signal that production is done


def process_frame(frame, frame_count):
    """Enhance a frame and perform human detection."""
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(equalized_image)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(f"frames/img_{frame_count}.jpg", enhanced_image)
    detect_human(enhanced_image, frame_count)
    print(f"Processed frame {frame_count}")


def process_video(cap, start_time, num_workers=4):
    """Consume frames from ``cap`` and process them with a worker pool."""
    print("Processing video")
    frame_queue = queue.Queue(maxsize=20)
    producer_executor = ThreadPoolExecutor(max_workers=1)
    producer_executor.submit(producer, cap, frame_queue, start_time)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while True:
            frame_data = frame_queue.get()
            if frame_data is None:
                break
            executor.submit(process_frame, *frame_data)


# Function to test video capture on a given index
def test_video_device(index):
    """Return ``True`` if a camera device can be opened."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open video device {index}.")
        return False
    else:
        print(f"Video device {index} opened successfully.")
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from device {index}.")
        cap.release()
        return True


def capture():
    """Capture frames from the global ``cap`` and process them."""
    # cam_index = 0
    # # Test both devices
    # for index in range(2):
    #     if test_video_device(index):
    #         cam_index = index
    #         break

    # cap = cv2.VideoCapture(cam_index)

    # Set video frame dimensions (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Camera connected")
    start_time = time.time()
    process_video(cap, start_time, num_workers=4)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time} s")
    cap.release()


def initialize():
    """Initialize GPIO pins and load the detection model."""
    global model, cap

    t1 = time.time()
    # Load the model from the local directory
    local_model_dir = "ssd_mobilenet_v2"
    model = hub.load(local_model_dir)

    t2 = time.time()
    print(f"SSD loaded in {t2-t1}s")

    input_pin = 17
    prev_value = None
    capturing = False
    cap = None

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(input_pin, GPIO.IN)

    if os.path.exists("frames"):
        shutil.rmtree("frames")
        shutil.rmtree("images")
        shutil.rmtree("faces")

    os.makedirs("frames", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    os.makedirs("faces", exist_ok=True)
    print("Starting surveillance")
    try:
        while True:
            value = GPIO.input(input_pin)
            if value != prev_value:
                if value == GPIO.LOW and not capturing:
                    print("Starting video capture")
                    cap = cv2.VideoCapture(0)

                    capturing = True
                    print("camera connected")
                    ThreadPoolExecutor(max_workers=1).submit(capture)
                elif value == GPIO.HIGH and capturing:
                    print("Stopping video capture")
                    if cap:
                        cap.release()
                    capturing = False
                prev_value = value
            time.sleep(1)

    except Exception as e:
        print(e)
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    initialize()
