# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
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

start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(
    "models/research/object_detection/test_data/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/"
)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: " + str(elapsed_time) + "s")

import cv2
import os
import numpy as np
from object_detection.utils import visualization_utils as viz_utils


# Assuming you have a detect_human function
def detect_human(image_np):
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    elapsed_time = end_time - start_time

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Filter and keep only 'person' class detections
    person_indices = np.where(
        (detections["detection_classes"][0].numpy() == category_index[1]["id"])
        & (detections["detection_scores"][0].numpy() > 0.5)
    )
    person_boxes = detections["detection_boxes"][0].numpy()[person_indices]
    person_classes = detections["detection_classes"][0].numpy()[person_indices]
    person_scores = detections["detection_scores"][0].numpy()[person_indices]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        person_boxes,
        person_classes.astype(np.int32),
        person_scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.40,
        agnostic_mode=False,
    )

    return image_np_with_detections


import cv2

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))


def enhance_video(input_video_path, output_video_path):
    # Read the video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(fps, frame_count)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    i = 0
    while i < 5:
        ret, frame = cap.read()

        if not ret:
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        equalized_image = cv2.equalizeHist(gray_img)

        enhanced_image = clahe.apply(equalized_image)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

        # Assuming you have 'detect_fn' and 'category_index' defined somewhere
        result_image = detect_human(enhanced_image)

        if i < 5:
            print(i)
            axes[i, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[i, 1].imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
            axes[i, 2].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB))
            axes[i, 3].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))

        # Enhance the frame

        # Write the frame into the file
        # out.write(result_image)

        i += 1

    # Release everything
    cap.release()
    out.release()


fig, axes = plt.subplots(5, 4, figsize=(8, 10))

# Example usage
input_video_path = "IMG_4514.MOV"
output_video_path = "IMG_4514_enhanced.mp4"
start_time = time.time()
enhance_video(input_video_path, output_video_path)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: " + str(elapsed_time) + "s")
# Ensure the layout is properly managed
plt.tight_layout()

# Display the plot
plt.show()
