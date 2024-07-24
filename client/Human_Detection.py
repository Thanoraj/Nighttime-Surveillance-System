import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

local_model_dir = "ssd_mobilenet_v2"
model = hub.load(local_model_dir)

def detect_human(image_path, frame_count):
    image = Image.open(image_path)
    image_np = np.array(image)

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
            cropped_image = Image.fromarray(cropped_image_np)
            cropped_image_path = f"images/cropped_image{frame_count}_{count}.jpg"
            cropped_image.save(cropped_image_path)
            count += 1

if __name__ == "__main__":
    detect_human("IMG_3913.jpg",1)