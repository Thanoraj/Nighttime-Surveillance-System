from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Directory where uploaded images will be saved
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    image_raw_bytes = request.get_data()  # get the whole body

    save_location = os.path.join(
        app.root_path, "static/test.jpg"
    )  # save to the same folder as the flask app live in

    f = open(
        save_location, "wb"
    )  # wb for write byte data in the file instead of string
    f.write(image_raw_bytes)  # write the bytes from the request body to the file
    f.close()

    print("Image saved")
    return (
        jsonify({"message": "File uploaded successfully", "filename": "test.jpg"}),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


# curl -X POST -H "Content-Type: image/jpeg" --data-binary @cal.jpg http://172.20.10.4:5001/upload
