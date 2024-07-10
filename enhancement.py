import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


data_dir = "datasets"

folders = os.listdir(f"{data_dir}/face_lowlight")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))

for folder in folders:
    print(folder)
    images = os.listdir(f"{data_dir}/face_lowlight/{folder}")
    if not os.path.exists(f"{data_dir}/face_enhanced/{folder}"):
        os.makedirs(f"{data_dir}/face_enhanced/{folder}")

    for image_name in images:
        image = cv2.imread(f"{data_dir}/face_lowlight/{folder}/{image_name}")

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # histogram_orig = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).ravel()

        # axes[0,0].imshow(image)
        # axes[0,1].plot(histogram_orig)

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_img)
        # histogram_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256]).ravel()

        # axes[0,1].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR))
        # axes[0,3].plot(histogram_equalized)

        enhanced_image = clahe.apply(equalized_image)
        # histogram_enhanced = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256]).ravel()
        # enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

        # axes[1,0].imshow(enhanced_image)
        # axes[1,1].plot(histogram_enhanced)

        # clahe_image = clahe.apply(gray_img)
        # histogram_clahe = cv2.calcHist([clahe_image], [0], None, [256], [0, 256]).ravel()
        # clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
        # axes[1,1].imshow(clahe_image)
        # axes[1,3].plot(histogram_clahe)

        # name = images[i].split(".")[0]

        # if not os.path.exists(f"{data_dir}/face_enhanced"):
        #   os.makedirs(f"{data_dir}/enhanced_clahe")
        #   os.makedirs(f"{data_dir}/enhanced_cega")
        #   os.makedirs(f"{data_dir}/enhanced_cega_clahe")
        cv2.imwrite(f"{data_dir}/face_enhanced/{folder}/{image_name}", equalized_image)
        # cv2.imwrite(f"{data_dir}/enhanced_cega_clahe/{name}.jpg", enhanced_image)
        # cv2.imwrite(f"{data_dir}/enhanced_clahe/{name}.jpg", clahe_image)

    # Ensure the layout is properly managed
    # plt.tight_layout()

    # Display the plot
    # plt.show()
