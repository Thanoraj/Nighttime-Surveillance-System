import datetime
import os
from threading import Thread
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from shutil import copy2
from PIL import Image
import numpy as np
from mtcnn import MTCNN


from firebase_services import (
    save_url_to_firestore,
    send_notification,
    upload_image,
)


class DeepFacePrediction:
    def __init__(self, embedding_path="embeddings.pkl"):
        self.predicting = False
        self.detector = MTCNN()

        self.false_positive = 0

        self.load_embeddings(embedding_path)
        self.sent = False
        print("Model loaded")

    # Initialize MTCNN detector

    # Function to extract face from an image
    def extract_face(self, filename, required_size=(160, 160)):
        image = Image.open(filename)
        image = image.convert("RGB")
        pixels = np.asarray(image)
        results = self.detector.detect_faces(pixels)
        if results:
            x1, y1, width, height = results[0]["box"]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            # image = image.resize(required_size)
            image.save(filename)
            # face_array = np.asarray(image)
            return filename
        else:
            self.false_positive += 1
            os.makedirs("false_positive", exist_ok=True)

            image.save(f"false_positive/img{self.false_positive}.jpg")

            return None

    # Load embeddings from a file
    def load_embeddings(self, filename):
        print(filename)
        with open(filename, "rb") as file:
            self.embeddings = pickle.load(file)
        return self.embeddings

    def sendAlert(self, image_path):
        filename = os.path.basename(image_path)

        destination_blob_name = f"images/HodSqk9AoxhgG3SfgWVL/{filename}"
        document_path = (
            f"Users/HodSqk9AoxhgG3SfgWVL/intruders/{os.path.splitext(filename)[0]}"
        )

        # Upload the image and get the URL
        image_url = upload_image(image_path, destination_blob_name)

        # Get the current date and time
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        print("Current date and time:", formatted_time)

        # Save the URL in Firestore
        save_url_to_firestore(image_url, formatted_time, document_path)

        send_notification("HodSqk9AoxhgG3SfgWVL", os.path.splitext(filename)[0])

    # Compare input image with known classes
    def identify_person(
        self, input_image, model_name="VGG-Face", metric="cosine", threshold=0.4
    ):

        if not self.extract_face(input_image):
            return None, None, None

        input_embedding = DeepFace.represent(
            img_path=input_image, model_name=model_name, enforce_detection=False
        )
        input_embedding = input_embedding[0]["embedding"]

        best_similarity = float("-inf")
        best_person = None

        intruders = {}

        for person, avg_embedding in self.embeddings.items():
            similarity = cosine_similarity([input_embedding], [avg_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person

        if best_similarity > threshold:
            os.makedirs("Home dwellers", exist_ok=True)
            intruder_image_path = os.path.join(
                "Home dwellers", f"{best_person}_{os.path.basename(input_image)}"
            )
            copy2(input_image, intruder_image_path)
            return best_person, input_embedding, best_similarity
        else:
            # Save the image of the intruder
            os.makedirs("Intruders", exist_ok=True)
            intruder_image_path = os.path.join(
                "Intruders", os.path.basename(input_image)
            )
            copy2(input_image, intruder_image_path)
            if not self.sent:
                Thread(target=self.sendAlert, args=(intruder_image_path)).start()
                self.sent = True
            return "Intruder", input_embedding, best_similarity

    def make_prediction(self):
        self.predicting = True
        images = os.listdir("uploads")

        # for i in range(len(images)):
        #     img = os.path.join("uploads", images[i])

        while len(images) != 0:
            img = os.path.join("uploads", images[0])
            prediction, _, _ = self.identify_person(img)
            print(f"Prediction for {images[0]} is {prediction}")
            os.remove(img)
            images = os.listdir("uploads")

        print(self.false_positive)
        self.predicting = False


if __name__ == "__main__":
    deepFacePrediction = DeepFacePrediction()
    images = os.listdir("uploads")
    results = {}
    for img in images:

        if img == ".DS_Store":
            continue

        res, _, _ = deepFacePrediction.identify_person(f"uploads/{img}")

        if res in results:
            results[res] += 1
        else:
            results[res] = 1

    print(results)
