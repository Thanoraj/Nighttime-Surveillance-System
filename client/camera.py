import cv2
import time

# Open a connection to the camera
cap = cv2.VideoCapture(0)  # '0' is typically the index for the first camera

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Set video frame dimensions (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for MP4 files
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    out.write(frame)  # Write the frame to the file

    # Break the loop after 5 seconds
    if time.time() - start_time > 60:
        break

# Release everything when done
cap.release()
out.release()
