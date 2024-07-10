from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Lambda,
)
from keras import backend as K
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# Define the convolutional neural network model for feature extraction
def create_feature_extractor_model():
    inputs = Input((64, 64, 1))

    # Convolutional layers
    x = Conv2D(96, (11, 11), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(384, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Fully connected layers
    pooled_output = GlobalAveragePooling2D()(x)
    pooled_output = Dense(1024)(pooled_output)
    outputs = Dense(128)(pooled_output)

    model = Model(inputs, outputs)
    return model


# Create the feature extraction model
feature_extractor_model = create_feature_extractor_model()

# Define the inputs for the final model
image_A = Input(shape=(64, 64, 1))
image_B = Input(shape=(64, 64, 1))

# Extract features for the two images
feature_A = feature_extractor_model(image_A)
feature_B = feature_extractor_model(image_B)


# Define the Euclidean distance function
def calculate_euclidean_distance(vectors):
    (feature_A, feature_B) = vectors
    sum_of_squares = K.sum(K.square(feature_A - feature_B), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_of_squares, K.epsilon()))


# Compute the Euclidean distance between the two feature vectors
euclidean_distance = Lambda(calculate_euclidean_distance)([feature_A, feature_B])

# Add the final layer and create the model
outputs = Dense(1, activation="sigmoid")(euclidean_distance)
final_model = Model(inputs=[image_A, image_B], outputs=outputs)

# Compile the model
final_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Define the function for generating image pairs for training
def generate_train_image_pairs(image_dataset, label_dataset):
    # Create a dictionary to store indices for each label
    unique_labels = np.unique(label_dataset)
    label_wise_indices = {
        label: np.where(label_dataset == label)[0] for label in unique_labels
    }

    # Generate image pairs and labels
    pair_images = []
    pair_labels = []

    for i, image in enumerate(image_dataset):
        # Generate a positive pair
        pos_indices = label_wise_indices[label_dataset[i]]
        pos_image = image_dataset[np.random.choice(pos_indices)]
        pair_images.append([image, pos_image])
        pair_labels.append(1)

        # Generate a negative pair
        neg_indices = np.where(label_dataset != label_dataset[i])[0]
        neg_image = image_dataset[np.random.choice(neg_indices)]
        pair_images.append([image, neg_image])
        pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels)


# Load the olivetti dataset
olivetti = datasets.fetch_olivetti_faces()

# Get the images and the labels
images_dataset = olivetti.images
labels_dataset = olivetti.target

# Generate the training image pairs
train_image_pairs, train_labels = generate_train_image_pairs(
    images_dataset, labels_dataset
)

# Fit the model
history = final_model.fit(
    [train_image_pairs[:, 0], train_image_pairs[:, 1]],
    train_labels,
    validation_split=0.1,
    batch_size=64,
    epochs=100,
)

# Select a random image as the test image
random_test_image = images_dataset[92]


def generate_test_image_pairs(images_dataset, labels_dataset, image):
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(
            label,
            [
                index
                for index, curr_label in enumerate(labels_dataset)
                if label == curr_label
            ],
        )

    pair_images = []
    pair_labels = []
    for label, indices_for_label in label_wise_indices.items():
        test_image = images_dataset[np.random.choice(indices_for_label)]
        pair_images.append((image, test_image))
        pair_labels.append(label)
    return np.array(pair_images), np.array(pair_labels)


# Generate the test image pairs
test_image_pairs, test_label_pairs = generate_test_image_pairs(
    images_dataset, labels_dataset, random_test_image
)

# Predict the similarity between the images in each pair
for i, pair in enumerate(test_image_pairs):
    pair_image1 = np.expand_dims(pair[0], axis=0)
    pair_image2 = np.expand_dims(pair[1], axis=0)
    similarity_prediction = final_model.predict([pair_image1, pair_image2])[0][0]
