"""Create darkened copies of images for data augmentation."""

import cv2
import numpy as np
import os
import random


# fig, axes = plt.subplots(1, 4, figsize=(12, 12))

i = 0


def convert_to_low_light(image_path, output_path, brightness_factor=0.1):
    global i
    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image")
        return

    image = image.astype(np.float32)

    low_light_image = image * brightness_factor

    low_light_image = np.clip(low_light_image, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, low_light_image)

    # axes[i].imshow(low_light_image)
    # axes[i].set_title(f"low_light with alpha={brightness_factor}")

    # print(f"Low light image saved to {output_path}")
    i += 1


folders = os.listdir("datasets/faces")

for folder in folders:
    frames = os.listdir(f"datasets/faces/{folder}")
    os.makedirs(f"datasets/face_enhanced/{folder}")
    for frame in frames:
        convert_to_low_light(
            f"datasets/faces/{folder}/{frame}",
            f"datasets/face_enhanced/{folder}/{frame}",
            round(random.uniform(0.1, 0.5), 2),
        )
