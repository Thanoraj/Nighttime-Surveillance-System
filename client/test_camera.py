import cv2


# Function to test video capture on a given index
def test_video_device(index):
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


# Test both devices
for index in range(2):
    test_video_device(index)
