
import os
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Step 1: Prepare the Dataset
embeddings_filename = '../face_recognition/embeddings.pkl'

# Load embeddings from a file
def load_embeddings(filename):
    with open(filename, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

# Compare input image with known classes
def identify_person(input_image, model_name='VGG-Face', metric='cosine', threshold=0.4):
    input_embedding = DeepFace.represent(img_path=input_image, model_name=model_name, enforce_detection=False)
    input_embedding = input_embedding[0]["embedding"]

    best_similarity = float('-inf')
    best_person = None
    for person, avg_embedding in class_embeddings.items():
        similarity = cosine_similarity([input_embedding], [avg_embedding])[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_person = person

    if best_similarity > threshold:
        return best_person, input_embedding, best_similarity
    else:
        return "Intruder", input_embedding, best_similarity


"""Inference"""

# Main Execution
print("Loading embeddings from file...")
class_embeddings = load_embeddings(embeddings_filename)

# Example Usage
input_image_path = "103.jpg"
model_name='VGG-Face'
metric='cosine'
threshold=0.4
result = identify_person(input_image_path, model_name, metric, threshold)
print(result)



class DeepFacePrediction:
    def __init__ (self, embedding_path=None):
        self.load_embeddings(embedding_path)
        return "Model Loaded"
    
    # Load embeddings from a file
    def load_embeddings(self, filename="embeddings.pkl"):
        with open(filename, 'rb') as file:
            self.embeddings = pickle.load(file)
        return self.embeddings
    
    # Compare input image with known classes
    def identify_person(self, input_image, model_name='VGG-Face', metric='cosine', threshold=0.4):
        input_embedding = DeepFace.represent(img_path=input_image, model_name=model_name, enforce_detection=False)
        input_embedding = input_embedding[0]["embedding"]

        best_similarity = float('-inf')
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
    deepFacePrediction.identify_person("")