from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import os

from PIL import Image
from huggingface_hub import hf_hub_download
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = hf_hub_download(
    repo_id="Priya-S-14/plant-disease-model",
    filename="plant_disease_recog_model_pwp.tflite",
    repo_type="model"
)

JSON_PATH = os.path.join(BASE_DIR, "plant_disease.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploadimages")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(JSON_PATH, "r") as file:
    plant_disease = json.load(file)


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((160, 160))
    feature = np.array(image, dtype=np.float32)
    feature = np.expand_dims(feature, axis=0)
    return feature


def model_predict(image_path):
    img = extract_features(image_path)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])

    prediction_index = int(np.argmax(prediction))
    prediction_label = plant_disease[prediction_index]

    return prediction_label


@app.route("/uploadimages/<path:filename>")
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/upload/", methods=["POST", "GET"])
def uploadimage():
    if request.method == "POST":
        if "img" not in request.files:
            return redirect(url_for("home"))

        image = request.files["img"]

        if image.filename == "":
            return redirect(url_for("home"))

        unique_name = f"temp_{uuid.uuid4().hex}_{image.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_name)

        image.save(image_path)

        print(f"Uploaded image: {image_path}")

        prediction = model_predict(image_path)

        image_url = url_for(
            "uploaded_images",
            filename=unique_name
        )

        return render_template(
            "home.html",
            result=True,
            imagepath=image_url,
            prediction=prediction
        )

    else:
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)