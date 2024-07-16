from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Hello world!"


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "image" not in request.files:
            print("No image part")
            return "No image part", 400
        file = request.files["image"]
        if file.filename == "":
            print("No selected file")
            return "No selected file", 400
        print(file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save("uploads/" + file.filename)
        return jsonify({"success": "File successfully uploaded"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": f"{e}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
