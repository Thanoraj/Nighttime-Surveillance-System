# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

# # Clone the tensorflow models repository if it doesn't already exist
# if "models" in pathlib.Path.cwd().parts:
#   while "models" in pathlib.Path.cwd().parts:
#     os.chdir('..')
# elif not pathlib.Path('models').exists():
#   !git clone --depth 1 https://github.com/tensorflow/models

# # Commented out IPython magic to ensure Python compatibility.
# # # Install the Object Detection API
# # %%bash
# # cd models/research/
# # protoc object_detection/protos/*.proto --python_out=.
# # cp object_detection/packages/tf2/setup.py .
# # python -m pip install .

import io
import scipy.misc
import six
import time

from six import BytesIO

from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, "rb").read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# Load the COCO Label Map
category_index = {1: {"id": 1, "name": "person"}}

# Download the saved model and put it into models/research/object_detection/test_data/
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(
    "models/research/object_detection/test_data/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/"
)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: " + str(elapsed_time) + "s")

"""Storing cropped human images"""

import cv2
import os
import numpy as np
import time
from object_detection.utils import visualization_utils as viz_utils
from concurrent.futures import ThreadPoolExecutor


def detect_human(image_np):
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()

    label_id_offset = 1
    person_indices = np.nonzero(
        (detections["detection_classes"][0].numpy() == category_index[1]["id"])
        & (detections["detection_scores"][0].numpy() > 0.7)
    )
    person_boxes = detections["detection_boxes"][0].numpy()[person_indices]
    return person_boxes


def process_frame(frame, frame_count, output_dir, frame_width, frame_height):
    # Convert frame to grayscale and enhance
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(equalized_image)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    # Detect humans
    person_boxes = detect_human(enhanced_image)

    # Crop and save the detected humans
    for i, (ymin, xmin, ymax, xmax) in enumerate(person_boxes):
        ymin, xmin, ymax, xmax = (
            int(ymin * frame_height),
            int(xmin * frame_width),
            int(ymax * frame_height),
            int(xmax * frame_width),
        )
        human = enhanced_image[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(output_dir, f"human_{frame_count}_{i}.jpg"), human)


def process_video(input_video_path, output_dir, num_workers=4):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    frames_to_process = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:  # Process every 5th frame
            process_frame(frame, frame_count, output_dir, frame_width, frame_height)
        frame_count += 1

    # Process frames in parallel

    cap.release()


# Example usage
input_video_path = (
    "/content/drive/Shareddrives/fyp/Datasets/Custom Dataset/IMG_8435.MOV"
)
output_dir = "/content/drive/Shareddrives/fyp/Datasets/humans_new"
os.makedirs(output_dir, exist_ok=True)
start_time = time.time()
process_video(input_video_path, output_dir, num_workers=4)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

import cv2
import os
import numpy as np
import time
from google.colab.patches import cv2_imshow
from object_detection.utils import visualization_utils as viz_utils
from concurrent.futures import ThreadPoolExecutor
import queue


def detect_human(image_np):
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()

    label_id_offset = 1
    person_indices = np.where(
        (detections["detection_classes"][0].numpy() == category_index[1]["id"])
        & (detections["detection_scores"][0].numpy() > 0.7)
    )
    person_boxes = detections["detection_boxes"][0].numpy()[person_indices]
    return person_boxes


def process_frame(data):
    frame, frame_count, output_dir, frame_width, frame_height = data
    # Convert frame to grayscale and enhance
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(equalized_image)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    # Detect humans
    person_boxes = detect_human(enhanced_image)

    # Crop and save the detected humans
    for i, (ymin, xmin, ymax, xmax) in enumerate(person_boxes):
        ymin, xmin, ymax, xmax = (
            int(ymin * frame_height),
            int(xmin * frame_width),
            int(ymax * frame_height),
            int(xmax * frame_width),
        )
        human = enhanced_image[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(output_dir, f"human_{frame_count}_{i}.jpg"), human)


def producer(cap, frame_queue, every_n_frame=10):
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n_frame == 0:
            frame_queue.put((frame, frame_count))
        frame_count += 1
    frame_queue.put(None)  # Signal that production is done


def process_video(input_video_path, output_dir, num_workers=4):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(output_dir, exist_ok=True)

    frame_queue = queue.Queue(maxsize=20)  # Adjust maxsize as needed

    # Start the producer thread
    producer_executor = ThreadPoolExecutor(max_workers=1)
    producer_future = producer_executor.submit(producer, cap, frame_queue)

    # Start consumer threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while True:
            frame_data = frame_queue.get()
            if frame_data is None:  # End signal
                break
            data = (*frame_data, output_dir, frame_width, frame_height)
            executor.submit(process_frame, data)

    cap.release()


# Example usage
input_video_path = (
    "/content/drive/Shareddrives/fyp/Datasets/Custom Dataset/IMG_8435.MOV"
)
output_dir = "/content/drive/Shareddrives/fyp/Datasets/humans_new"
start_time = time.time()
process_video(input_video_path, output_dir, num_workers=4)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

"""Storing cropped face images from the detected human images


"""

import cv2
import os
import numpy as np
import time
from google.colab.patches import cv2_imshow


# Function to detect faces using OpenCV's Haar Cascade classifier
def detect_faces(image):
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30)
    )

    return faces


# Function to process human images, detect faces, crop faces, and save cropped face images
def process_human_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Iterate through each human image file in the input directory
    for filename in os.listdir(input_dir):

        if filename.endswith(".jpg"):
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Detect faces in the image
            faces = detect_faces(image)

            # Crop and save the detected faces
            for i, (x, y, w, h) in enumerate(faces):
                face = image[y : y + h, x : x + w]

                cv2.imwrite(
                    os.path.join(output_dir, f'{filename.split(".")[0]}_face_{i}.jpg'),
                    face,
                )


# Example usage
input_dir = "/content/drive/Shareddrives/fyp/Datasets/humans_new"  # Directory containing extracted human images
output_dir = "/content/drive/Shareddrives/fyp/Datasets/faces_new"  # Directory to store cropped face images
os.makedirs(output_dir, exist_ok=True)
process_human_images(input_dir, output_dir)

"""Aspect ratio calculation"""

import cv2
import numpy as np


# Assuming you have a detect_human function
def detect_human(image_np):
    # Replace this with your actual detection code
    # For example, using SSD MobileNet V2
    input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # Filter and keep only 'person' class detections
    person_indices = np.where(
        (detections["detection_classes"][0].numpy() == category_index[1]["id"])
        & (detections["detection_scores"][0].numpy() > 0.7)
    )
    person_boxes = detections["detection_boxes"][0].numpy()[person_indices]

    return person_boxes


def calculate_height_width_ratio(bbox):
    # Calculate height and width from bounding box coordinates
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]

    # Calculate the ratio of height to width
    ratio = height / width

    return ratio


# Example usage
image = cv2.imread(
    "/content/drive/Shareddrives/fyp/Datasets/humans_new/human_107_0.jpg"
)  # Replace with your image
person_boxes = detect_human(image)

if len(person_boxes) > 0:
    bbox = person_boxes[0]  # Assume the first detected person
    ratio = calculate_height_width_ratio(bbox)
    print("Height-Width Ratio:", ratio)
else:
    print("No person detected.")

"""Calculating Aspect Ratio of Family Members"""

import os
from PIL import Image


# Function to load images from a folder with labels
def load_images_from_folder(folder_path):
    images = {}
    for label in os.listdir(folder_path):
        print(label)
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            images[label] = []
            for filename in os.listdir(label_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(label_path, filename)
                    images[label].append(image_path)
    return images


# Function to calculate height-to-width ratio from annotated images
def calculate_ratio(image_path):
    # Load the annotated image
    annotated_image = Image.open(image_path)
    person_boxes = detect_human(
        annotated_image
    )  # Assuming you have a function to detect humans in the image
    if len(person_boxes) > 0:
        bbox = person_boxes[0]  # Assume the first detected person
        ratio = calculate_height_width_ratio(bbox)
        return ratio
    return None


# Function to calculate average ratios for each family member
def calculate_average_ratios(images):
    average_ratios = {}
    for label, image_paths in images.items():
        ratios = []
        for image_path in image_paths:
            ratio = calculate_ratio(image_path)
            if ratio is not None:  # Filter out None values
                ratios.append(ratio)
        if ratios:
            average_ratio = sum(ratios) / len(ratios)
            average_ratios[label] = average_ratio
        else:
            # If no ratios were calculated for the label, set the average ratio to None
            average_ratios[label] = None
    return average_ratios


# Example usage:
# Assuming images of family members are stored in a folder 'family_images' with labels as subfolders
folder_path = "/content/drive/Shareddrives/fyp/Datasets/Family"
images = load_images_from_folder(folder_path)
average_ratios = calculate_average_ratios(images)

print(average_ratios)

"""
import os
from PIL import Image

# Function to load images from a folder with labels
def load_images_from_folder(folder_path):
    images = {}
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            images[label] = []
            for filename in os.listdir(label_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(label_path, filename)
                    images[label].append(image_path)
    return images

# Function to calculate height-to-width ratio from annotated images
def calculate_ratio(image_path):
    # Load the annotated image
    annotated_image = Image.open(image_path)
    person_boxes = detect_human(image)
    # Extract height and width from annotation (you need to implement this part)
    height = image['height']
    width = image['width']
    # Calculate ratio
    ratio = height / width
    return ratio

# Function to calculate average ratios for each family member
def calculate_average_ratios(images):
    average_ratios = {}
    for label, image_paths in images.items():
        ratios = []
        for image_path in image_paths:
            ratio = calculate_ratio(image_path)
            ratios.append(ratio)
        if ratios:
            average_ratio = sum(ratios) / len(ratios)
            average_ratios[label] = average_ratio
    return average_ratios

# Example usage:
# Assuming images of family members are stored in a folder 'family_images' with labels as subfolders
folder_path = 'family_images'
images = load_images_from_folder(folder_path)
average_ratios = calculate_average_ratios(images)
"""

"""Comparing detected person's ratio with the known ratios"""

# Step 1: Define Known Ratios
family_ratios = {
    "father": 1.8,  # Example height-to-width ratio for father
    "mother": 1.6,  # Example height-to-width ratio for mother
    "child": 1.3,  # Example height-to-width ratio for child
}


# Step 2: Calculate Ratio for Detected Person
def calculate_ratio(bounding_box):
    # Calculate height and width from bounding box coordinates
    height = bounding_box[2] - bounding_box[0]
    width = bounding_box[3] - bounding_box[1]
    # Calculate height-to-width ratio
    ratio = height / width
    return ratio


# Step 3: Compare Ratios
def compare_ratios(detected_ratio):
    min_similarity = float("inf")
    closest_match = None
    for member, known_ratio in family_ratios.items():
        similarity = abs(detected_ratio - known_ratio)
        if similarity < min_similarity:
            min_similarity = similarity
            closest_match = member
    return closest_match, min_similarity


# Step 4: Threshold for Intruder Detection
threshold = 0.1  # Adjust as needed based on the similarity metric

# Example usage:
# Assuming 'detected_box' contains the bounding box coordinates of the detected person
image = cv2.imread(
    "/content/drive/Shareddrives/fyp/Datasets/humans_new/human_107_0.jpg"
)  # Replace with your image
person_boxes = detect_human(image)

if len(person_boxes) > 0:
    bbox = person_boxes[0]  # Assume the first detected person
    detected_ratio = calculate_height_width_ratio(bbox)
closest_member, similarity = compare_ratios(detected_ratio)

print(similarity)
if similarity < threshold:
    print(f"The detected person closely matches {closest_member}.")
else:
    print("Potential intruder detected!")

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

# Load the original image
original_image = cv2.imread(
    "/content/drive/Shareddrives/fyp/Datasets/Custom Dataset/enhanced_faces/Pranna/IMG_3940_face_0.jpg"
)

# Add Gaussian noise to the image
mean = 0
variance = 100
sigma = variance**0.5
gaussian_noise = np.random.normal(mean, sigma, original_image.shape).astype(np.uint8)
noisy_image = cv2.add(original_image, gaussian_noise)

# Compute PSNR between original and noisy images
psnr = peak_signal_noise_ratio(original_image, noisy_image)

print("PSNR:", psnr)

# Plot original and noisy images as subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot original image
axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")
axs[0].axis("off")

# Plot noisy image
axs[1].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
axs[1].set_title("Noisy Image")
axs[1].axis("off")

plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

# Load the original image
original_image = cv2.imread(
    "/content/drive/Shareddrives/fyp/Datasets/Custom Dataset/enhanced_faces/Pranna/IMG_3940_face_0.jpg"
)

# Define the initial variance and increment/decrement value
initial_variance = 100
variance_step = 10

# Define the number of iterations
num_iterations = 9

# Initialize the variance value
variance = initial_variance

# Create a figure to display the subplots
fig, axs = plt.subplots(1, num_iterations + 1, figsize=(15, 5))

# Plot the original image
axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")
axs[0].axis("off")

# Loop through the iterations
for i in range(1, num_iterations + 1):
    # Add Gaussian noise to the image
    noisy_image = cv2.add(
        original_image,
        np.random.normal(0, variance, original_image.shape).astype(np.uint8),
    )

    # Compute PSNR between original and noisy images
    psnr = peak_signal_noise_ratio(original_image, noisy_image)

    # Save the noisy image with PSNR value in the filename
    cv2.imwrite(f"noisy_image_psnr_{psnr:.2f}.jpg", noisy_image)

    # Plot the noisy image
    axs[i].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    axs[i].set_title(f"Noisy Image (PSNR: {psnr:.2f})")
    axs[i].axis("off")

    print(f"Noisy image saved with PSNR {psnr:.2f}")

    # Update the variance for the next iteration
    variance -= variance_step

    # Ensure variance is non-negative
    if variance < 0:
        break

plt.tight_layout()
plt.show()
