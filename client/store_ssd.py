import tensorflow as tf
import tensorflow_hub as hub
import os


def install_ssd():
    # Directory to save the model
    local_model_dir = "ssd_mobilenet_v2"

    # Download and save the model
    os.makedirs(local_model_dir)
    model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    tf.saved_model.save(model, local_model_dir)
    print(f"Model downloaded and saved to {local_model_dir}")
