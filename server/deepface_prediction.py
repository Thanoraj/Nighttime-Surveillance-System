import datetime
import os
from threading import Thread
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from shutil import copy2

from server.firebase_services import (
    save_url_to_firestore,
    send_notification,
    upload_image,
)


class DeepFacePrediction:
    def __init__(self, embedding_path="embeddings.pkl"):
        self.predicting = False
        self.load_embeddings(embedding_path)
        print("Model loaded")

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
            return best_person, input_embedding, best_similarity
        else:
            # Save the image of the intruder
            os.makedirs("Intruders", exist_ok=True)
            intruder_image_path = os.path.join(
                "Intruders", os.path.basename(input_image)
            )
            copy2(input_image, intruder_image_path)
            Thread(target=self.sendAlert, args=(intruder_image_path)).start()
            return "Intruder", input_embedding, best_similarity

    def make_prediction(self):
        self.predicting = True
        images = os.listdir("uploads")

        while len(images) != 0:
            img = os.path.join("uploads", images[0])
            prediction, _, _ = self.identify_person(img)
            print(f"Prediction for {images[0]} is {prediction}")
            os.remove(img)
            images = os.listdir("uploads")

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
