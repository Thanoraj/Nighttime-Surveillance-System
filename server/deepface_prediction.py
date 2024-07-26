import os
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class DeepFacePrediction:
    def __init__(self, embedding_path="embeddings.pkl"):
        self.load_embeddings(embedding_path)
        print("Model loaded")

    # Load embeddings from a file
    def load_embeddings(self, filename):
        print(filename)
        with open(filename, "rb") as file:
            self.embeddings = pickle.load(file)
        return self.embeddings

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
        for person, avg_embedding in self.embeddings.items():
            similarity = cosine_similarity([input_embedding], [avg_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person

        if best_similarity > threshold:
            return best_person, input_embedding, best_similarity
        else:
            return "Intruder", input_embedding, best_similarity


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
