"""Detect faces in images using OpenCV's Haar cascade."""

import cv2
import os


def detect_faces(image, frame_count, human_count):
    """Return bounding boxes for faces detected in ``image``."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=4, minSize=(10, 10)
    )

    print(len(faces))

    for i, (x, y, w, h) in enumerate(faces):
        face = image[y : y + h, x : x + w]
        image_path = os.path.join("faces", f"faces_{frame_count}_{human_count}_{i}.jpg")
        cv2.imwrite(image_path, face)
        # server_url = "http://192.168.8.157:5001/upload"
        # send_image(image_path, server_url)

    return faces


if __name__ == "__main__":

    humans = os.listdir("images")
    for index, human in enumerate(humans):
        image = cv2.imread(os.path.join("images", human))
        print(image)
        detect_faces(image, index, index)
