import socket
import os
from tensorflow.keras import layers, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import threading

# Server configuration
server_ip = "0.0.0.0"
server_port = 5001
buffer_size = 4096

# Setup server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)
print(f"Listening as {server_ip}:{server_port} ...")

# Setting random seeds to enable consistency while testing.
random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)

frames_left = 0
frames_lock = threading.Lock()


def get_encoder(input_shape):
    """Returns the image encoding model"""

    pretrained_model = Xception(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )
    print(len(pretrained_model.layers))

    for i in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential(
        [
            pretrained_model,
            layers.Flatten(),
            layers.Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
            layers.Dropout(0.5),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
        ],
        name="Encode_Model",
    )

    print(len(encode_model.layers))
    return encode_model


class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_siamese_network(input_shape=(128, 128, 3)):
    encoder = get_encoder(input_shape)

    # Input Layers for the images
    anchor_input = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input), encoder(positive_input), encoder(negative_input)
    )

    # Creating the Model
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network",
    )
    return siamese_network


siamese_network = get_siamese_network()
siamese_network.summary()


class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]


siamese_model = SiameseModel(siamese_network)

optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)

# Path to the saved weights
weights_path = "siamese_model"

# Load the previously saved weights
siamese_model.load_weights(weights_path)

print("Weights loaded successfully.")


def extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i = 0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i += 1
    return encoder


encoder = extract_encoder(siamese_model)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image1 = cv2.resize(image, (128, 128))  # Resize image
    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image1 = preprocess_input(
        image1
    )  # Assuming you have the preprocess_input function as used during training
    return image1


def check_intruder(path):
    image_path1 = path
    image1 = preprocess_image(image_path1)
    embedding1 = encoder.predict(image1)

    embedding_path = "Dataset"
    folders = os.listdir(embedding_path)

    average_distances = {}

    for i, folder in enumerate(folders, start=0):

        class_path = os.path.join(embedding_path, folder)
        files = os.listdir(class_path)

        distances = []

        for j in range(5):
            random_file = random.choice(files)
            file_path = os.path.join(class_path, random_file)
            image2 = preprocess_image(file_path)

            embedding2 = encoder.predict(image2)
            distance = np.linalg.norm(embedding1 - embedding2)
            distances.append(distance)

        # Calculate the average distance for the current class
        average_distance = np.mean(distances)
        average_distances[folder] = average_distance

    print(average_distances)

    # Determine if the minimum average distance is less than 1.5
    if min(average_distances.values()) < 1:
        result = False
    else:
        result = True

    print(f"{result} for {path}")


def synchronizer():
    global frames_left
    with frames_lock:
        while frames_left > 0:
            files = os.listdir("videos")
            path = os.path.join("videos", files[0])
            print(path)
            check_intruder(path)
            os.remove(path)
            frames_left -= 1


try:
    while True:
        client_socket, address = server_socket.accept()
        print(f"Connection from {address} has been established.")

        while True:
            # Receiving the file name length and the file name
            file_name_length_data = client_socket.recv(4)
            if not file_name_length_data:
                print("Client has closed the connection.")
                break

            file_name_length = int.from_bytes(file_name_length_data, "big")
            file_name = client_socket.recv(file_name_length).decode()

            # Receiving the file size
            file_size_data = client_socket.recv(8)
            if not file_size_data:
                print("No file size data received. Possible client disconnection.")
                break
            file_size = int.from_bytes(file_size_data, "big")

            os.makedirs("videos", exist_ok=True)

            # Receive and write the file
            with open(f"videos/{file_name}", "wb") as file:
                bytes_received = 0
                while bytes_received < file_size:
                    chunk = client_socket.recv(
                        min(buffer_size, file_size - bytes_received)
                    )
                    if not chunk:
                        break  # Connection closed
                    file.write(chunk)
                    bytes_received += len(chunk)

            if frames_left == 0:
                print(frames_left)
                with frames_lock:
                    frames_left += 1
                    background_thread = threading.Thread(target=synchronizer)
                    # Start the thread
                    background_thread.start()

            else:
                frames_left += 1

            print(f"File {file_name} received successfully.")

except KeyboardInterrupt:
    print("\nServer is shutting down.")
finally:
    server_socket.close()
    print("Server socket closed.")
